"""Free the GPU: stop run 1 now, or after the current step (see `src/sqlrl/run1/pause.py`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.run1.pause import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
