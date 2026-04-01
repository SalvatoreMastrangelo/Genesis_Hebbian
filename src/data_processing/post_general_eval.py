from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from post_general_eval_utils import URDFHistogramPlotter, main

__all__ = ["URDFHistogramPlotter", "main"]


if __name__ == "__main__":
    main()
