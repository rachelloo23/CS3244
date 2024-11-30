# Decision Tree Model

This directory contains the implementation and evaluation of the Decision Tree (DT) model. Four distinct approaches were implemented to study the impact of feature selection and oversampling techniques on the decision tree's effectiveness.

## Directory Structure

```plaintext
.
├── decision-tree.py               # Plain decision tree implementation
├── feature-selection.py           # Feature selection with decision tree
├── oversampling.py                # Oversampling with decision tree
├── feature_and_oversample.py      # Combines feature selection and oversampling
├── README.md                      # This README
```

## Files

### Python Scripts
- **`decision-tree.py`**: Baseline implementation of the Decision Tree model with standard scaling and cross-validation.
- **`feature-selection.py`**: Applies correlation-based feature selection to remove redundant features before training the Decision Tree model.
- **`oversampling.py`**: Uses SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance before training.
- **`feature_and_oversample.py`**: Combines feature selection and oversampling techniques to enhance the Decision Tree model.

## Reproducing Results
Run `.py` to reproduce the result.
```bash
python <filename>.py
```

## Results Summary

| **Approach**                 | **F1 Score (Train)** | **F1 Score (Test)** |
|------------------------------|----------------------|---------------------|
| **Plain Decision Tree**      | 98.12%              | 82.80%             |
| **Feature Selection**        | 97.07%              | 83.21%             |
| **Oversampling**             | 100%                | 83.10%             |
| **Feature Selection + Oversampling** | 97.28%       | 84.25%             |

---
