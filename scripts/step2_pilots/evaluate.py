"""Evaluate a finished pilot (src/sqlrl/step2_pilots/evaluate.py). Windows Python."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.step2_pilots.evaluate import main  # noqa: E402

sys.exit(main())
