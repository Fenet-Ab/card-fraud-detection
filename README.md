# Credit Card Fraud Detection

This project studies credit card fraud detection as an imbalanced binary
classification problem.

The target column is `Class`:

- `0` means a legitimate transaction.
- `1` means a fraudulent transaction.

The project compares several ways to handle the class imbalance:

- Baseline Gradient Boosting model
- SMOTE oversampling
- Random undersampling
- Class-weighted training
- Threshold tuning for an alert budget

## Dataset

The dataset is already included in this repo:

- Raw data: `data/raw/creditcard.csv`
- Processed data: `data/processed/creditcard_processed.csv`

The processed file currently has `283,726` transactions and `31` columns.
The class distribution is:

| Class | Meaning | Count |
| --- | --- | ---: |
| `0` | Legitimate | 283,253 |
| `1` | Fraud | 473 |

This imbalance is the main reason the project compares SMOTE,
undersampling, class weighting, and threshold tuning.

## Project Structure

```text
card-fraud-detection/
|-- data/
|   |-- raw/creditcard.csv
|   `-- processed/creditcard_processed.csv
|-- models/
|   |-- baseline_model.pkl
|   |-- smote_model.pkl
|   |-- undersampling_model.pkl
|   `-- class_weight_model.pkl
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_baseline_model.ipynb
|   |-- 03_smote.ipynb
|   |-- 04_undersampling.ipynb
|   |-- 05_class_weight.ipynb
|   |-- 06_threshold_tuning.ipynb
|   `-- 07_model_comparison.ipynb
|-- results/
|   `-- model_comparison.csv
|-- src/
|   |-- data/preprocess.py
|   |-- evaluation/
|   |-- imbalance/
|   `-- models/
|-- train.py
`-- requirements.txt
```

## Setup

From the project folder, create and activate a virtual environment:

```bash
cd /home/fenet/Documents/card-fraud-detection
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## How To Run The Notebooks

Start Jupyter:

```bash
jupyter lab
```

Then open the notebooks folder and run the notebooks in this order:

1. `01_data_exploration.ipynb`
2. `02_baseline_model.ipynb`
3. `03_smote.ipynb`
4. `04_undersampling.ipynb`
5. `05_class_weight.ipynb`
6. `06_threshold_tuning.ipynb`
7. `07_model_comparison.ipynb`

Run each notebook from top to bottom.

## Where To See The Effect In The Notebooks

Use these notebook cells to clearly see what each step changes.

| Notebook | Where to look | What you will see |
| --- | --- | --- |
| `01_data_exploration.ipynb` | Cells `7`, `11`, and `12` | The class imbalance between legitimate and fraud transactions. |
| `01_data_exploration.ipynb` | Cell `14` | Transaction amount distribution by class. |
| `01_data_exploration.ipynb` | Cell `16` | The processed CSV saved to `data/processed/creditcard_processed.csv`. |
| `02_baseline_model.ipynb` | Cells `9` to `13` | Baseline classification report, precision, recall, F1, PR-AUC, and confusion matrix. |
| `02_baseline_model.ipynb` | Cell `14` | Baseline model saved to `models/baseline_model.pkl`. |
| `03_smote.ipynb` | Cells `4` to `6` | Class distribution before and after SMOTE. This is the clearest place to see the oversampling effect. |
| `03_smote.ipynb` | Cells `11` to `15` | SMOTE model metrics and confusion matrix. |
| `04_undersampling.ipynb` | Cells `5` to `8` | Class distribution before and after undersampling. This shows how legitimate transactions are reduced in the training set. |
| `04_undersampling.ipynb` | Cells `13` to `18` | Undersampling metrics, confusion matrix, and precision-recall curve. |
| `05_class_weight.ipynb` | Cell `5` | The calculated sample weights for legitimate and fraud transactions. |
| `05_class_weight.ipynb` | Cells `8` to `13` | Class-weighted model metrics, confusion matrix, and precision-recall curve. |
| `06_threshold_tuning.ipynb` | Cells `11` to `18` | Threshold results table, best threshold under the alert budget, and default-vs-tuned comparison. |
| `06_threshold_tuning.ipynb` | Cells `13` to `15` | Plots showing how precision, recall, F1, and alerts change as the threshold changes. |
| `07_model_comparison.ipynb` | Cell `14` | Final comparison table for Baseline, SMOTE, Undersampling, and Class Weight. |
| `07_model_comparison.ipynb` | Cells `17` to `21` | Best precision, best recall, best F1, and comparison charts. |
| `07_model_comparison.ipynb` | Cell `16` | Saves the final comparison to `results/model_comparison.csv`. |

## Run Everything From The Script

You can also run the pipeline from the command line:

```bash
python train.py
```

This loads `data/processed/creditcard_processed.csv`, trains/evaluates the
models using the code in `src/`, prints the final comparison table, and saves:

```text
results/src_model_comparison.csv
```

The script version also includes a `Threshold Tuning` row.

## Current Saved Results

The current notebook comparison in `results/model_comparison.csv` is:

| Method | Precision | Recall | F1 | PR-AUC | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.8611 | 0.6526 | 0.7425 | 0.6234 | 72 |
| SMOTE | 0.1951 | 0.8316 | 0.3160 | 0.7242 | 405 |
| Undersampling | 0.0443 | 0.8632 | 0.0842 | 0.5846 | 1,853 |
| Class Weight | 0.1850 | 0.8316 | 0.3027 | 0.7004 | 427 |

Interpretation:

- The baseline model has the highest precision and F1 in the saved results.
- Undersampling has the highest recall, so it catches more fraud cases, but it
  also creates many more false alerts.
- SMOTE and class weighting increase recall compared with the baseline, but
  precision drops because they flag more transactions as fraud.
- Threshold tuning lets you control the number of fraud alerts by changing the
  probability cutoff.

## Outputs

Important generated outputs are:

- Trained models: `models/*.pkl`
- Notebook comparison: `results/model_comparison.csv`
- Script comparison: `results/src_model_comparison.csv`

