"""PyInstaller entry: no arguments opens the UI; explicit arguments use the CLI."""
import multiprocessing
import sys

from exploration.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main(sys.argv[1:] or ["ui"])
