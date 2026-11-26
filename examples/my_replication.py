"""Full dataset/task/model LabTOP pipeline entry point for the project rubric."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.labtop.train import main

if __name__ == "__main__":
    main()
