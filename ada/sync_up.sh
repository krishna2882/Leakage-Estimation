#!/bin/bash
# Push code (not outputs) to Ada.  Run from the project folder on the laptop.
G=krishna.goel@ada-gw1.iiit.ac.in
rsync -az --exclude .git --exclude __pycache__ --exclude 'data*' --exclude 'results*' --exclude 'plots*' \
      --exclude 'saved_models*' --exclude 'work*' --exclude logs ./ $G:projects/dvd/project/
