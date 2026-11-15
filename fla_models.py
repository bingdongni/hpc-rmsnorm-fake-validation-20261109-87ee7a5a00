import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

here = Path(__file__).resolve().parent
root = Path(r"C:\PR")
output = Path(os.environ.get("RUNNER_TEMP", str(here)))
checkouts = [Path(x).resolve() for x in sys.argv[1:]] if len(sys.argv) > 1 else [root / "fla-kda-baseline-20261110", here / "fla-model-head"]
results = []
metrics = []
for label, checkout, sha in [("base", checkouts[0], "b8da4873fa0834849b64938af9f89bc35a347d90"),
                             ("candidate", checkouts[1], "9ac47b38bd6a6e6cd4071c05d3b5dc544ea4114d")]:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == sha
    assert not subprocess.check_output(["git", "diff", "--name-only"], cwd=checkout)
    env = os.environ.copy()
    env.update(FLA_CI_ENV="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", HF_HUB_OFFLINE="1", PYTHONUTF8="1",
               CC=str(root / "fla-research-deps-20261110/triton/runtime/tcc/tcc.exe"),
               CUDA_PATH=str(root / "fla-research-deps-20261110/triton/backends/nvidia"),
               PYTHONPATH=os.pathsep.join(map(str, (checkout, root / "fla-research-deps-20261110",
                                                       root / "fla-model-deps-20261110"))))
    env.pop("FLA_DISABLE_BACKEND_DISPATCH", None)
    subprocess.run([sys.executable, "-c", "from pathlib import Path;import fla,torch,triton; "
                    "assert Path(fla.__file__).resolve().is_relative_to(Path.cwd()); "
                    "assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.__version__, triton.__version__)"],
                   cwd=checkout, env=env, check=True)
    xml = output / ("fla-model-" + label + ".xml")
    with (output / ("fla-model-" + label + ".log")).open("w", encoding="utf8") as log:
        check = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "log_cli=false", "--disable-warnings",
                                "--tb=short", "--junitxml", str(xml), "tests/models/test_modeling_kda.py"],
                                cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    assert check.returncode in (0, 1)
    cases = ET.parse(xml).findall(".//testcase")
    item = {c.attrib["name"]: next((k for k in ("failure", "skipped", "error") if c.find(k) is not None), "pass") for c in cases}
    assert len(item) == 15
    results.append(item)
    errors = {}
    for c in cases:
        failure = c.find("failure")
        if failure is not None:
            magnitude = re.search(r"output diff:\s*([0-9.]+) ratio:\s*([0-9.]+)", failure.get("message") or "")
            assert magnitude and c.attrib["name"].endswith("-varlen]")
            errors[c.attrib["name"]] = magnitude.groups()
    metrics.append(errors)
    print(label, {s: list(item.values()).count(s) for s in set(item.values())}, flush=True)
    for name, pair in sorted(errors.items()):
        print(name, "output diff", pair[0], "ratio", pair[1], flush=True)
    (output / "fla-model-results.json").write_text(json.dumps(results, indent=2), encoding="utf8")
assert results[0] == results[1], "Candidate and base model suites differ"
assert metrics[0] == metrics[1] and all(x in ("pass", "failure") for x in results[0].values())
