#!/bin/bash
# Mirror Ada outputs back to the laptop (outputs on Ada are the source of truth).
G=krishna.goel@ada-gw1.iiit.ac.in
for d in data results plots saved_models netlists logs; do
  rsync -az --delete $G:projects/dvd/project/$d/ ./$d/ 2>/dev/null
done
# dataset-size runs: results and plots of every run; data and models only of the headline run (common.HEADLINE)
for t in _2k5 _5k _20k _40k; do
  for d in results plots; do rsync -az --delete $G:projects/dvd/project/$d$t/ ./$d$t/ 2>/dev/null; done
done
for d in data saved_models; do rsync -az --delete $G:projects/dvd/project/${d}_40k/ ./${d}_40k/ 2>/dev/null; done
