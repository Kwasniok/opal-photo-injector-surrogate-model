#!/usr/bin/env bash

set -euo pipefail

echo "Checking installation..."
./install.sh > /dev/null

source venv/bin/activate
jupyter lab