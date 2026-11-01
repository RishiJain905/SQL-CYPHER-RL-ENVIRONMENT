"""Run a pilot (src/sqlrl/step2_pilots/orchestrate.py). Windows Python, standard library."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.step2_pilots.orchestrate import main  # noqa: E402

sys.exit(main())
