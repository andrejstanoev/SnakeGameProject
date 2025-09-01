# SnakeGameProject

A minimalist Snake game solved with a genetic algorithm using PyGAD. The GA evolves a weight vector that maps game state features to actions, improving over generations to eat apples efficiently. Run the main script to train and then automatically test the best-found agent.

## Features
- Genetic Algorithm training with PyGAD, including configurable generations, population size, mutation rate, and gene space bounds. 
- Compact state representation with 24 features capturing collisions, distances, directions, food layout, and board sparsity.  
- Reproducible training loop via fitness evaluation averaged across multiple episodes and capped steps.  

## Project structure
- scripts/main.py – Entry point; runs GA, prints best fitness, and executes a testing rollout. Run this to start and finish.
- scripts/snake.py – Environment and fitness function; defines game mechanics, state extraction, and reward structure.
- scripts/constants.py – All key hyperparameters and board configuration; edit values here to tune GA behavior.
- scripts/utils.py – Utility functions for distance computation and action selection from weights.

## Requirements
Python 3.9+ recommended.\
PyGAD: pip install pygad.

## How to run
From the project root, run the main script to start training and automatically test the best agent at the end:
- python scripts/main.py

The script will:
- Build a GA with parameters from scripts/constants.py.
- Train for the configured number of generations.
- Print the best solution’s fitness.
- Launch a test run that logs steps, actions, rewards, score, snake positions, and remaining apples until termination.

## Tuning the algorithm
All tunable parameters live in scripts/constants.py. To change how “good” the genetic algorithm performs, edit values starting at NUM_FOOD and below. These influence difficulty, exploration, and convergence. After editing, rerun main.

Key parameters to consider:
- NUM_FOOD: Number of apples on the grid; more apples can shape reward frequency and learning signals.
- NUM_EPISODES: Episodes per fitness evaluation; higher averages reduce noise but slow training.
- MAX_STEPS: Step cap per episode; allows longer exploration at higher values.
- NUM_GENERATIONS: Total evolutionary iterations; more generations generally improve solutions at added time cost.
- NUM_PARENTS_MATING: Selection pressure; interacts with population size.
- SOL_PER_POP: Population size; larger pops explore more but cost more compute.
- MUTATION_PERCENT_GENES: Controls mutation strength; too low may stagnate, too high may disrupt convergence.
- GENE_SPACE_LOW/HIGH: Value bounds for genes (weights); constrains search region for stability.

Tip: Start by modestly increasing NUM_GENERATIONS and SOL_PER_POP, and adjust MUTATION_PERCENT_GENES if progress plateaus. Keep NUM_EPISODES modest to balance stability and speed.

## How it works
- State: 24 features including collision danger, wall distances, body proximity, relative food direction, direction one-hots, apples statistics, and empty-cell fraction.
- Policy: A weight matrix shaped from the chromosome maps state to action scores; select argmax.
- Fitness: Average episode reward across NUM_EPISODES with step and terminal rewards encouraging survival and apple collection.
- GA loop: Selection, mutation, and evolution over NUM_GENERATIONS within specified gene space bounds.

## Troubleshooting
- Slow training: Lower NUM_EPISODES or MAX_STEPS, or reduce SOL_PER_POP while validating improvements.
- Unstable learning: Narrow GENE_SPACE bounds, increase NUM_GENERATIONS, or tune MUTATION_PERCENT_GENES.
- Environment differences: Ensure Python and PyGAD versions are compatible; reinstall in a fresh venv if needed.