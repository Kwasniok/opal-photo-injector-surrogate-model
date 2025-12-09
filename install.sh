#!/usr/bin/env bash

set -euo pipefail

# venv
python3 -m venv venv
source venv/bin/activate

# install
pip install --upgrade pip
pip install -r requirements.txt

# ipykernel
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
DISPLAY_NAME="$(basename "$(pwd)") (python${PYTHON_VERSION})"
# remove non-alphanumeric characters for kernel name
KERNEL_NAME=$(echo "$DISPLAY_NAME" | tr -cd '[:alnum:]')
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"
