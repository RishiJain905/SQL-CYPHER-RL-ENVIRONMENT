"""Run the sampling baseline (src/sqlrl/baseline/run.py). Windows Python, standard library."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlrl.baseline.run import main  # noqa: E402

sys.exit(main())
