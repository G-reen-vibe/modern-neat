# Modern NEAT: A Fundamental Rewrite of NeuroEvolution of Augmenting Topologies

This repository contains a research project exploring modern, fundamental
rewrites of the NEAT algorithm (Stanley & Miikkulainen, 2002) for reinforcement
learning control tasks such as CartPole, MountainCar, and Acrobot.

## Project Goals

1. Implement a clean, faithful baseline of original NEAT.
2. Implement research-standard baselines: fixed-topology GA, CMA-ES, random
   search, and (where tractable) DQN.
3. Iterate on a *fundamental* rewrite of NEAT — not a hybrid of two algorithms,
   but a new algorithm that unifies modern insights (gradient information,
   behavior-driven diversity, adaptive complexity) into a single elegant
   principle.
4. Evaluate every variant on multiple benchmarks with multiple seeds and report
   error bars / confidence intervals.
5. Produce a comprehensive report viewable as a live JS site.

## Repository Layout

```
modern-neat/
  src/           - core algorithm implementations
  experiments/   - experiment driver scripts
  results/       - experiment outputs (CSV, plots, logs)
  reports/       - final report and assets
```

## Research Process

The research follows a 4-phase protocol:
- Phase 0: Setup + baselines + evaluation framework
- Phase 1: 75 rounds of algorithmic iteration (with rethinks at 25/50/75)
- Phase 2: ~25 rounds of ablation and analysis on the chosen algorithm
- Phase 3: Comprehensive live report
