#!/usr/bin/bash

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)

git add --all
git commit -m "backup $TIMESTAMP"
git push -u origin main