#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1

# Finder launches may not inherit the terminal's PATH.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Please ask your instructor to complete setup."
    read -r -p "Press Enter to close."
    exit 1
fi

echo "Starting JupyterLab. Keep this window open while working."
uv run --locked jupyter lab
result=$?
if [ "$result" -ne 0 ]; then
    echo "JupyterLab exited with an error. Please show this window to your instructor."
    read -r -p "Press Enter to close."
fi
exit "$result"
