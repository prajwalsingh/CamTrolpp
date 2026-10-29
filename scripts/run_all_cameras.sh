#!/bin/bash
# Run every camera action on one image, each saved as its own run.
#   bash scripts/run_all_cameras.sh assets/examples/image000.png [outputs]
set -e
cd "$(dirname "$0")/.."
python run.py --image "$1" --config configs/*.yaml --out "${2:-outputs}" --skip-existing
