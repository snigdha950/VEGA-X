"""Judge-facing RAFTS TSRD demo."""
from pathlib import Path
import sys
from dataset_replay import main

if __name__ == "__main__":
    # Allows: python demo.py data/tsrd/config_103.h5
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        sys.argv = [sys.argv[0], "--file", sys.argv[1]]
    elif len(sys.argv) == 1:
        default = Path("data/tsrd/config_103.h5")
        if not default.exists():
            raise SystemExit("Put a TSRD .h5 file in data/tsrd/ or pass its path.")
        sys.argv = [sys.argv[0], "--file", str(default)]
    main()
