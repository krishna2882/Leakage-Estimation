#!/bin/bash
# Mirror Ada outputs back to the laptop (outputs on Ada are the source of truth).
G=krishna.goel@ada-gw1.iiit.ac.in
for d in data results plots saved_models netlists logs; do
  rsync -az --delete $G:projects/dvd/project/$d/ ./$d/ 2>/dev/null
done
