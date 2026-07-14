# Atlas-NEAT: Modern Quality-Diversity NeuroEvolution

A modern replacement for NEAT that replaces speciation with an adaptive Quality-Diversity archive.

## Core Innovation

Atlas-NEAT replaces NEAT's speciation mechanism with an **adaptive Quality-Diversity archive**.
Networks are mapped to archive cells based on their topological and behavioral characteristics,
maintaining diversity explicitly through the archive structure.

### Key Differences from NEAT
1. **No speciation** - diversity maintained by archive
2. **No compatibility distance** - archive placement based on auto-characterization
3. **No stagnation detection** - natural turnover in archive
4. **Novelty-guided exploration** - adaptive mutation rates
5. **Adaptive resolution** - archive grows as coverage increases

## Installation

```bash
pip install gymnasium neat-python matplotlib numpy scipy torch
```

## Running Experiments

```bash
# Run baseline comparison
python run_experiments.py --benchmarks cartpole --algorithms neat atlas --trials 5 --plot

# Run Atlas-NEAT only
python run_experiments.py --benchmarks cartpole lunarlander --algorithms atlas --trials 5 --plot
```

## Project Structure

```
src/
  atlas_neat/         # Atlas-NEAT implementation
    genome.py         # Genome representation
    archive.py        # QD archive
    atlas_neat.py     # Main algorithm
  baselines/          # Baseline implementations
    neat_baseline.py  # Standard NEAT
    neat_config.ini   # NEAT configuration
  benchmarks/         # Evaluation utilities
    evaluator.py      # Environment evaluation
results/              # Experimental results
figures/              # Generated plots
```
