# Mushroom-CNN-Comparison

Mushroom-CNN-Comparison benchmarks multiple convolutional neural network architectures on a mushroom image classification task and provides post-training analysis utilities.

## Project Objective
Compare model behavior across modern CNN backbones and produce interpretable evaluation outputs such as confusion matrices and training curves.

## Core Technologies
- Python 3
- PyTorch
- torchvision / timm-style model usage
- scikit-learn
- matplotlib

## Compared Model Families
- ConvNeXt
- EfficientNetB0
- MobileNetV3

## Project Workflow
1. Prepare and split dataset
2. Train each architecture
3. Log train/test metrics
4. Generate comparison visualizations

## Main Scripts
### Training
- `src/models/convnext.py`
- `src/models/efficientnetb0.py`
- `src/models/mobilenetv3.py`
- `src/models/logger.py`
- `src/models/split_code.py`

### Analysis
- `src/graphs/confusion_matrix.py`
- `src/graphs/graph.py`
- `src/graphs/data.py`

## Installation
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Typical Run Flow
```bash
python src/models/convnext.py
python src/models/efficientnetb0.py
python src/models/mobilenetv3.py
python src/graphs/confusion_matrix.py
python src/graphs/graph.py
```

## Outputs
- `train_log.json`
- `test_log.json`
- confusion matrix plots
- comparison graphs

## Repository Structure
```text
Mushroom-CNN-Comparison/
 src/models/
 src/graphs/
 train_log.json
 test_log.json
 requirements.txt
 README.md
```

## Notes
- Use fixed seeds and consistent splits for fair architecture comparison.
- GPU is strongly recommended for practical training times.
- Include class-balance checks to avoid misleading accuracy metrics.
