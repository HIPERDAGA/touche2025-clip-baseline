#!/bin/bash

set -e

INPUT_DIR=$1
OUTPUT_DIR=$2

python3 touche2025_v6_tira.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
