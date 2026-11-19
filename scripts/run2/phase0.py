"""Run 2's phase 0: the no-training checks (see `src/sqlrl/run2/phase0.py`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.run2.phase0 import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
