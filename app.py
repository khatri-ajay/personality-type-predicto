"""Streamlit app: answer 19 questions, get a personality type."""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st

from utils import EXPERIMENT_NAME, MLFLOW_URI

QUESTIONS = {
    "N1": "I get stressed out easily.",
    "N2": "I am relaxed most of the time.",
    "N3": "I worry about things.",
    "N4": "I seldom feel blue.",
    "N5": "I am easily disturbed.",
    "N6": "I get upset easily.",
    "N7": "I change my mood a lot.",
    "N8": "I have frequent mood swings.",
    "N9": "I get irritated easily.",
    "N10": "I often feel blue.",
    "E1": "I am the life of the party.",
    "E3": "I feel comfortable around people.",
    "E4": "I keep in the background.",
    "E5": "I start conversations.",
    "E7": "I talk to a lot of different people at parties.",
    "E9": "I don't mind being the center of attention.",
    "E10": "I am quiet around strangers.",
    "C4": "I make a mess of things.",
    "A4": "I sympathize with others' feelings.",
}

TYPE_INFO = {
    "Moderate": "Balanced, with no extreme traits.",
    "Resilient": "Emotionally stable and calm under pressure.",
    "Overcontroller": "Anxious and introverted.",
    "Undercontroller": "Impulsive and less concerned with rules.",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the best logged run (highest cross-validated F1-macro), or the run
    named in the MODEL_RUN_ID environment variable."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    run_id = os.environ.get("MODEL_RUN_ID")
    if not run_id:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                                  order_by=["metrics.f1_macro_cv DESC"], max_results=1)
        if runs.empty:
            raise RuntimeError("No runs found. Run notebook 03 first.")
        run_id = runs.iloc[0]["run_id"]
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model"), run_id


st.set_page_config(page_title="Personality type predictor")
st.title("Personality type predictor")
st.write("Rate how much you agree with each statement "
         "(1 = Disagree, 5 = Agree), then press Predict.")

try:
    model, run_id = load_model()
except Exception as exc:  # no database yet, or no runs
    st.error(f"Could not load a model: {exc}\n\n"
             "Run notebook 03 (MLflow tracking) first so a model is logged.")
    st.stop()

with st.form("questionnaire"):
    st.subheader("About you")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", min_value=13, max_value=100, value=25)
    gender = c2.selectbox("Gender", ["Female", "Male", "Other"])
    hand = c3.selectbox("Writing hand", ["Right", "Left", "Both"])

    st.subheader("Statements")
    answers = {code: st.slider(text, 1, 5, 3, key=code)
               for code, text in QUESTIONS.items()}
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {**answers, "age": age, "gender": gender, "hand": hand}
    # same columns, same names as training; the pipeline does the rest
    X_new = pd.DataFrame([row])[list(model.feature_names_in_)]
    prediction = model.predict(X_new)[0]
    st.success(f"Your predicted type: **{prediction}**")
    st.write(TYPE_INFO[prediction])

    proba = pd.Series(model.predict_proba(X_new)[0], index=model.classes_, name="probability")
    st.bar_chart(proba)
    st.caption(f"Model from MLflow run {run_id}")