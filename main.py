"""Launch the single MLSD/OCS page directly from this project directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mlsd_ocs_core.gui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
