"""Run 1, start to finish; also resumes (see `src/sqlrl/run1/orchestrate.py`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.run1.orchestrate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
