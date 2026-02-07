# Mushroom CNN Comparison

Compare multiple CNN architectures for mushroom species classification.
Includes model definitions, logging utilities, and analysis scripts.

## Structure

- src/models: model definitions and training helpers
- src/graphs: analysis utilities and plotting scripts

## Workflow

1. Split dataset (see `src/models/split_code.py`).
2. Train models and write logs (train_log.json, test_log.json).
3. Run analysis scripts to generate comparison plots and confusion matrices.

## Notes

- Training logs and model weights are generated locally and are not tracked in git.
- See requirements.txt for core dependencies.
