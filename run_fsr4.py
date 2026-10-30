"""Entry point for the packaged exe (PyInstaller cannot start a package's `__main__` directly)."""
import sys

from fsr4.cli import main

if __name__ == "__main__":
    sys.exit(main())
