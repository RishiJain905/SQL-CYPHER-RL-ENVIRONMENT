"""The baseline's dashboard feed (src/sqlrl/baseline/feed.py). Windows Python, standard library."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlrl.baseline.feed import main  # noqa: E402

sys.exit(main())
