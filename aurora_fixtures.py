"""Cache the published Microsoft regression fixtures at a verified Hugging Face revision."""
import hashlib
import os
from pathlib import Path
import sys
import time

import requests

REVISION = "a96afd7ee6d65e3bd2d476f3be798a25a56f2296"
FILES = {
    "aurora-0.25-small-pretrained-test-input.pickle": "f52e985b91e259de15176080f98a860383f7ef9641a6c1dedf04c62f37a487df",
    "aurora-0.25-small-pretrained-test-output.pickle": "e7ea0bd469632b3d4720d391a21b4bbe39bae14785f6e494547e9d6abb6cd453",
    "aurora-0.25-static.pickle": "e382103f6b24bcf1f996cc0af217c71ff2fc66507a5221e1300b5017581bd318",
}


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b""):
            checksum.update(block)
        return checksum.hexdigest()


def main(home):
    repo = home / "hub" / "models--microsoft--aurora"
    dest = repo / "snapshots" / REVISION
    dest.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        path = dest / name
        if not path.is_file() or digest(path) != sha:
            url = f"https://huggingface.co/microsoft/aurora/resolve/{REVISION}/{name}"
            for attempt in range(3):
                try:
                    with requests.get(url, params={"download": "true", "verify": str(time.time_ns())},
                                      stream=True, timeout=(15, 75)) as r:
                        r.raise_for_status()
                        with path.with_suffix(".part").open("wb") as output:
                            for chunk in r.iter_content(1048576):
                                output.write(chunk)
                    assert digest(path.with_suffix(".part")) == sha, name
                    os.replace(path.with_suffix(".part"), path)
                    break
                except (AssertionError, requests.RequestException):
                    if attempt == 2:
                        raise RuntimeError(f"Verified download failed for {name}") from None
            print(name, sha, flush=True)
    refs = repo / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(REVISION, encoding="ascii")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
