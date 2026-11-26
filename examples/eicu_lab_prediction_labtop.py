"""Rubric-named LabTOP full pipeline and held-out history ablation entry point.

Run from the repository root after installing the checkout with pip install -e .:
    python examples/eicu_lab_prediction_labtop.py --synthetic --epochs 2 \
        --max-new-tokens 8 --history-keep 1 .75 .5 --output /tmp/labtop_smoke

See examples/labtop/README.md for a longer demonstration and clinical commands.
"""

import sys
from pathlib import Path

# A direct examples/foo.py invocation otherwise omits the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.labtop.train import main

if __name__ == "__main__":
    main()
