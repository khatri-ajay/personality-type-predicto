# Personality Type Predictor

A model that predicts one of four personality types (Moderate, Resilient, Overcontroller, Undercontroller) from 19 questionnaire answers plus age, gender and writing hand. It is served by a Streamlit web app.

## Results

Scores come from 5-fold cross-validation on the training data, plus one final check on a held-out test set (F1-macro).

FILL IN: paste your results table here (from the end of notebook 02 or 03), for example:

| Model | CV F1-macro | Test F1-macro |
| --- | --- | --- |
| Hist gradient boosting (tuned) | 0.00 | 0.00 |

Undercontroller is the hardest type to predict. It depends on low Conscientiousness and low Agreeableness, and the dataset keeps only one question for each.

## Repo layout

```text
README.md                      this file
requirements.txt               pinned libraries
.gitignore                     files git should ignore
data.csv                       the dataset
utils.py                       data cleaning, preprocessing pipeline, model helper
best_params.json               tuned settings written by notebook 02
01_eda_and_pipeline.ipynb      explore the data, build the pipeline
02_evaluate_and_tune.ipynb     cross-validation, model comparison, Hyperopt tuning
03_mlflow_tracking.ipynb       log runs to MLflow and rank them
app.py                         Streamlit app that loads the best model
```

## Data

`data.csv` is a cleaned version of the Big Five (OCEAN) personality test dataset from Kaggle, prepared by the course. Each row is one person:

- 19 question scores (1 = Disagree to 5 = Agree)
- `age`, `gender`, `hand`
- `target`: the personality type to predict

Two cleaning rules are applied in `utils.py`: drop the row where all 19 answers are 0, and drop rows with an age outside 13 to 100.

## Setup and run

Tested with Python FILL IN (run `python --version`).

```bash
git clone FILL IN (your repo URL)
cd personality-project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open the project folder in VS Code or Jupyter and run each notebook from top to bottom, in this order:

1. `01_eda_and_pipeline.ipynb` (about 1 minute)
2. `02_evaluate_and_tune.ipynb` (about 6 minutes, writes `best_params.json`)
3. `03_mlflow_tracking.ipynb` (about 1 minute, logs the models)

Then, from the project folder:

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"   # optional: browse runs at http://127.0.0.1:5000
streamlit run app.py                                         # the web app
```

Run the MLflow UI and the app in two separate terminals.

## Limitations

The personality type is derived from all 50 questions of the original test, but only 19 are kept here, so the model can approximate the rule but not reproduce it exactly.
