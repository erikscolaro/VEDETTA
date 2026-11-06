#!/bin/bash
# Open Images V7 boxes (annotations CC BY 4.0, Google LLC; images listed as CC BY 2.0): files used, fetched once.
set -e
B=https://storage.googleapis.com/openimages
for u in v7/oidv7-class-descriptions-boxable.csv \
         v6/oidv6-train-annotations-bbox.csv v5/validation-annotations-bbox.csv v5/test-annotations-bbox.csv \
         2018_04/train/train-images-boxable-with-rotation.csv 2018_04/validation/validation-images-with-rotation.csv \
         2018_04/test/test-images-with-rotation.csv; do
  f=$(basename $u); [ -s "$f" ] || curl -sfSL -o "$f" "$B/$u"
done
sha256sum *.csv > sha256.txt
date -Iseconds > scaricato_il.txt
