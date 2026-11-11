import os
from pathlib import Path
import sys
import types

import pytest

root = Path(os.environ["TORCHFT_PROBE_ROOT"])
package = types.ModuleType("torchft")
package.__path__ = [str(root / "torchft")]
sys.modules["torchft"] = package
raise SystemExit(pytest.main([str(root / "torchft/collectives_work_test.py"), *sys.argv[1:]]))
