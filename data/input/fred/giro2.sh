#!/bin/bash
# secondo giro FRED (README, Procedura)
set -e
: "${DETECTOR_PESI:?serve DETECTOR_PESI per --step yolo}"
cd ~/Desktop/vedetta
S="2 0 198 209 111 108 133 165 19 28"
py() { .venv/bin/python tools/fred.py "$@" 2>&1 | grep -v "^download"; }
py --step frame --seq $S
py --step annotati --seq $S
py --step dedup --seq $S
py --step ritaglio --seq $S
py --step yolo --seq 108 111
py --step movimento --seq 2 0 198 209 111 133 165 19 28
py --step fogli_yolo --seq 108 111
py --step fogli_movimento --seq 2 0 198 209 111 133 165 19 28
