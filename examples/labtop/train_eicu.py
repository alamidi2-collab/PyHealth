"""Compatibility entry point for the LabTOP eICU example."""

from examples.labtop.train import evaluate, main, train

__all__ = ["evaluate", "main", "train"]

if __name__ == "__main__":
    main()
