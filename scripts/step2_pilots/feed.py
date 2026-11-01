"""The pilot tracker's feed (src/sqlrl/step2_pilots/feed.py). Windows Python, standard library."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.step2_pilots.feed import main  # noqa: E402

sys.exit(main())
