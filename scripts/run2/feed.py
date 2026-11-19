"""One tick of the dashboard's data for run 2 (see `src/sqlrl/run2/feed.py`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.run2.feed import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
