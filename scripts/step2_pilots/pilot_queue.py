"""Run pilots one after another (src/sqlrl/step2_pilots/pilot_queue.py). Windows Python."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.step2_pilots.pilot_queue import main  # noqa: E402

sys.exit(main())
