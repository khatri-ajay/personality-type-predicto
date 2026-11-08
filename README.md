# Personality Type Predictor

A model that predicts one of four personality types (Moderate, Resilient, Overcontroller, Undercontroller) from 19 questionnaire answers plus age, gender and writing hand. It is served by a Streamlit web app.

## Results

All models use `class_weight="balanced"` in the tuned versions, and are scored with F1-macro, which weights the four classes equally. This matters because the classes are imbalanced.

**Untuned model comparison** (5-fold cross-validation on the full cleaned data):

| Model | CV F1-macro | Std between folds |
| --- | --- | --- |
| Hist gradient boosting (balanced) | 0.8058 | 0.0065 |
| Hist gradient boosting | 0.7986 | 0.0082 |
| Random forest (balanced) | 0.7857 | 0.0070 |
| Logistic regression (balanced) | 0.7838 | 0.0109 |
| Logistic regression | 0.7664 | 0.0094 |
| Random forest | 0.7424 | 0.0074 |
| KNN | 0.7207 | 0.0084 |

**Tuned models** (Hyperopt on the training split, then one final check on the held-out 20% test set):

| Model | Test F1-macro | Test accuracy |
| --- | --- | --- |
| Hist gradient boosting | 0.8031 | 0.8365 |
| Random forest | 0.7844 | 0.8189 |
| Logistic regression | 0.7797 | 0.8235 |

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
hello.py                       small Streamlit warm-up example
pin_versions.py                prints library versions for requirements.txt
```

## Data

`data.csv` is a cleaned version of the Big Five (OCEAN) personality test dataset from Kaggle, prepared by the course. Each row is one person:

- 19 question scores (1 = Disagree to 5 = Agree)
- `age`, `gender`, `hand`
- `target`: the personality type to predict

Two cleaning rules are applied in `utils.py`: drop the row where all 19 answers are 0, and drop rows with an age outside 13 to 100.

## Setup and run

Tested with Python 3.14.3.

```bash
git clone https://github.com/khatri-ajay/personality-type-predicto.git
cd personality-type-predicto
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open the project folder in VS Code or Jupyter and run each notebook from top to bottom, in this order:

1. `01_eda_and_pipeline.ipynb` (about 1 minute)
2. `02_evaluate_and_tune.ipynb` (about 5 to 10 minutes, writes `best_params.json`)
3. `03_mlflow_tracking.ipynb` (about 1 minute, logs the models)

Then, from the project folder:

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"   # optional: browse runs at http://127.0.0.1:5000
streamlit run app.py                                         # the web app
```

Run the MLflow UI and the app in two separate terminals. The app loads the run with the best cross-validated F1-macro from MLflow, so notebook 03 must be run first.

## Limitations

The personality type is derived from all 50 questions of the original test, but only 19 are kept here, so the model can approximate the rule but not reproduce it exactly.