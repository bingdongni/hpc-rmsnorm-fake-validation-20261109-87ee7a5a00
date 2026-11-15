"""Compare the complete GPU state-update suite on two separately imported checkouts."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

TEST = Path("tests/ops/triton/test_selective_state_update.py")
EXPECTED = {
    "test_selective_state_update[2064-64-True-itype2]",
    "test_selective_state_update_with_heads[2048-64-2-True-True-itype2]",
    "test_selective_state_update_with_batch_indices[4096-64-True-itype2]",
}


def suite(root, expected_count, directory):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    identify = ("from pathlib import Path; import mamba_ssm,torch,triton; "
                "assert Path(mamba_ssm.__file__).resolve().is_relative_to(Path.cwd()); "
                "assert torch.cuda.is_available(); print(torch.cuda.get_device_name(),torch.__version__,triton.__version__)")
    subprocess.run([sys.executable, "-c", identify], cwd=root, env=env, check=True)
    output = directory / (root.name + ".xml")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short", str(TEST),
                             "--junitxml", str(output)], cwd=root, env=env)
    assert result.returncode in (0, 1)
    cases = ET.parse(output).findall(".//testcase")
    assert len(cases) == expected_count and not any(x.find("error") is not None or
                                                     x.find("skipped") is not None for x in cases)
    names = {x.attrib["name"] for x in cases}
    failures = {x.attrib["name"] for x in cases if x.find("failure") is not None}
    print(root.name, expected_count - len(failures), "passed", sorted(failures), flush=True)
    return names, failures


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_default(base, candidate):
    import torch
    tests = load("gpu_state_tests", candidate / TEST)
    old = load("gpu_state_base", base / "mamba_ssm/ops/triton/selective_state_update.py")
    new_update = tests.selective_state_update
    matched = 0

    def paired(state, *args, **kwargs):
        nonlocal matched
        old_state = state.clone()
        expected = old.selective_state_update(old_state, *args, **kwargs)
        actual = new_update(state, *args, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, old_state, rtol=0, atol=0)
        matched += 1
        return actual

    tests.selective_state_update = paired
    known = [
        (tests.test_selective_state_update, (2064, 64, True, torch.bfloat16)),
        (tests.test_selective_state_update_with_heads, (2048, 64, 2, True, True, torch.bfloat16)),
        (tests.test_selective_state_update_with_batch_indices, (4096, 64, True, torch.bfloat16)),
    ]
    for n, (fn, args) in enumerate(known, 1):
        try:
            fn(*args)
        except AssertionError:
            if matched != n:
                raise
    assert matched == len(known)
    print("Default output and updated cache are bitwise equal across checkouts in all three BF16 cases.", flush=True)


def main(base, candidate):
    with tempfile.TemporaryDirectory() as directory:
        base_names, baseline = suite(base, 540, Path(directory))
        head_names, failures = suite(candidate, 544, Path(directory))
        assert baseline <= EXPECTED and failures == baseline
        assert base_names <= head_names
        assert len(head_names - base_names) == 4 and all(
            name.startswith("test_indexed_state_writeback[") for name in head_names - base_names)
        assert not (failures & (head_names - base_names))
    sys.path.insert(0, str(candidate))
    compare_default(base, candidate)


if __name__ == "__main__":
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))
