"""Launch the Streamlit application entry point."""

from __future__ import annotations

import os
import subprocess
import sys


def main(argv: list[str] | None = None) -> None:
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    command = ["streamlit", "run", script_path]
    if argv:
        command.extend(argv)

    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Streamlit CLI not found. Ensure Streamlit is installed and available in PATH."
        ) from exc

    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main(sys.argv[1:])

