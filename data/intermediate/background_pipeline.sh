#!/bin/bash
# Passo 13: background da Clouds-1500 e Skyfinder (tools/background.py), tutte le fasi in ordine.
set -e
cd /home/scolaroe/Desktop/vedetta
for f in clouds1500 skyfinder; do
  for s in ritagli detector dedup fogli; do
    echo "=== $f $s $(date +%T)"
    .venv/bin/python -u tools/background.py --fonte $f --step $s 2>&1 | grep -v "xFormers\|warnings.warn\|Using cache"
  done
done
echo "=== split $(date +%T)"
.venv/bin/python -u tools/background.py --step split
echo "=== FINITO $(date +%T)"
