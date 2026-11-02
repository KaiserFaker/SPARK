"""SPARK dashboard, a Streamlit app for scoring students with the frozen models."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from spark_core import RISK_ORDER, assign_risk, load_bundle, load_portuguese, score_students

MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "outputs" / "tables"
MAIN_MODEL = "Random Forest"
BORDERLINE_GRADES = [9, 10, 13, 14]

st.set_page_config(page_title="SPARK", layout="wide")


# ===== Loading helpers =====
@st.cache_resource
def get_bundle(path):
    return load_bundle(path)


@st.cache_data
def get_table(path):
    return pd.read_csv(path)


@st.cache_data
def get_demo_cohort():
    return load_portuguese()[1]


def available_models():
    files = sorted(MODELS_DIR.glob("v2_bundle_*.joblib"))
    return {f.stem.replace("v2_bundle_", "").replace("_", " "): f for f in files}


def show_risk(risk):
    if risk == "High":
        st.error("High risk")
    elif risk == "Medium":
        st.warning("Medium risk")
    else:
        st.success("Low risk")

# ===== Suggestions for lecturers =====
SHORT_ACTIONS = {
    "High": "Meet within two weeks and agree a catch up plan",
    "Medium": "Check in and share revision guidance",
    "Low": "No action needed",
}


def suggest_actions(risk, stage, g1, g2, borderline):
    """Rule based guidance using only the risk label, the stage and the grades."""
    if risk == "Low":
        return []

    steps = []
    if risk == "High":
        steps.append("Arrange a one to one meeting within the next two weeks to talk about the "
                     "result and what is getting in the way.")
        steps.append("Review the topics where marks were lost and agree a short catch up plan "
                     "with clear weekly targets.")
        steps.append("Point the student to tutoring, peer study groups or academic advising.")
        if stage == "stage1":
            steps.append("Only the first grade is available, so confirm the concern with the "
                         "next assessment before deciding on stronger action.")
    else:
        steps.append("Check in briefly with the student and go through the most recent assessment.")
        steps.append("Share revision guidance for the next assessment and invite the student "
                     "to consultation hours.")

    if pd.notna(g2):
        change = g2 - g1
        if change <= -2:
            steps.append("The grade fell between G1 and G2, so ask what changed in that period, "
                         "such as workload, attendance or understanding.")
        elif change >= 2:
            steps.append("The grade improved between G1 and G2, so recognise the progress and "
                         "keep the current support going.")
    if borderline:
        steps.append("The grade is next to a band boundary, so a small gain could change the "
                     "outcome and a short push may help.")
    return steps

# ===== Tab 1, single student =====
def single_tab(bundle):
    left, right = st.columns([1, 2])

    with left:
        g1 = st.number_input("G1, first period grade (0 to 20)",
                             min_value=0, max_value=20, value=10, step=1)
        has_g2 = st.checkbox("G2 is available", value=True)
        g2 = st.number_input("G2, second period grade (0 to 20)",
                             min_value=0, max_value=20, value=10, step=1,
                             disabled=not has_g2)

    row = pd.DataFrame({"G1": [g1], "G2": [g2 if has_g2 else np.nan]})
    result = score_students(row, bundle).iloc[0]

    stage = result["stage"]
    threshold = bundle["thresholds"][stage]
    probs = pd.Series({"High": float(result["p_high"]),
                       "Medium": float(result["p_medium"]),
                       "Low": float(result["p_low"])})
    latest = g2 if has_g2 else g1

    with right:
        show_risk(result["risk"])
        st.write("Stage used: " + ("Stage 2, G1 and G2" if stage == "stage2" else "Stage 1, G1 only"))
        st.bar_chart(probs)
        st.write(f"A student is flagged High when the probability of High is at least {threshold:.3f}.")
        st.write(f"A simple rule that applies the bands to the latest grade gives: {assign_risk(latest)}.")
        if latest in BORDERLINE_GRADES:
            st.info("This grade sits next to a band boundary, so treat the label as uncertain.")
        steps = suggest_actions(result["risk"], stage, g1, g2 if has_g2 else np.nan,
                                latest in BORDERLINE_GRADES)
        if steps:
            st.subheader("Suggested actions")
            st.markdown("\n".join("- " + step for step in steps))
            st.caption("General guidance based on the risk label and the grades, not a finding "
                       "of the model and not tested for effectiveness. Use your own judgement "
                       "and talk to the student.")
        else:
            st.caption("No action suggested for Low risk students.")


# ===== Tab 2, cohort =====
def cohort_tab(bundle):
    source = st.radio("Data source",
                      ["Upload a CSV file", "Demo cohort, Portuguese students"],
                      horizontal=True)

    if source == "Upload a CSV file":
        uploaded = st.file_uploader("CSV file with a G1 column and, optionally, a G2 column", type="csv")
        if uploaded is None:
            st.info("Upload a file to see the results.")
            return
        raw = pd.read_csv(uploaded, sep=None, engine="python")
    else:
        raw = get_demo_cohort()
        st.caption("Demo only. These students were not used to train the model, and their final "
                   "grades are included so the predictions can be checked.")

    output_cols = ["stage", "p_high", "p_medium", "p_low", "predicted_risk",
                   "borderline", "_latest", "suggested_action"]
    data = (raw.rename(columns={"risk": "actual_risk"})
               .drop(columns=output_cols, errors="ignore")
               .reset_index(drop=True))

    try:
        scored = score_students(data, bundle)
    except ValueError as error:
        st.error(str(error))
        return

    scored = scored.rename(columns={"risk": "predicted_risk"})
    for col in ["p_high", "p_medium", "p_low"]:
        scored[col] = pd.to_numeric(scored[col])

    result = pd.concat([data, scored], axis=1)
    result["G1"] = pd.to_numeric(result["G1"])
    if "G2" in result.columns:
        result["G2"] = pd.to_numeric(result["G2"])
        latest = result["G2"].fillna(result["G1"])
    else:
        latest = result["G1"]
    if "G3" in result.columns:
        result["actual_risk"] = result["G3"].apply(assign_risk)

    result["borderline"] = np.where(latest.isin(BORDERLINE_GRADES), "Yes", "No")
    result["suggested_action"] = result["predicted_risk"].map(SHORT_ACTIONS)
    result["_latest"] = latest

    # Filters
    c1, c2, c3 = st.columns(3)
    risk_filter = c1.multiselect("Predicted risk", RISK_ORDER, default=RISK_ORDER)
    stages = sorted(result["stage"].dropna().unique())
    stage_filter = c2.multiselect("Stage", stages, default=stages)
    only_border = c3.checkbox("Borderline students only")

    filtered = result[result["predicted_risk"].isin(risk_filter) & result["stage"].isin(stage_filter)]
    for col in ["school", "sex", "address"]:
        if col in filtered.columns:
            options = sorted(result[col].dropna().unique())
            picked = st.multiselect(col.capitalize(), options, default=options)
            filtered = filtered[filtered[col].isin(picked)]
    if only_border:
        filtered = filtered[filtered["borderline"] == "Yes"]

    # Summary
    flagged = int((filtered["predicted_risk"] == "High").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students shown", len(filtered))
    m2.metric("Flagged High", flagged)
    m3.metric("Share flagged", f"{flagged / max(len(filtered), 1):.0%}")
    m4.metric("Borderline", int((filtered["borderline"] == "Yes").sum()))

    counts = filtered["predicted_risk"].value_counts().reindex(RISK_ORDER).fillna(0)
    st.bar_chart(counts)

    # Ranked table, ties broken by the lower latest grade, then the lower G1
    ranked = filtered.sort_values(["p_high", "_latest", "G1"], ascending=[False, True, True])
    show_cols = [c for c in ["school", "sex", "address", "G1", "G2", "stage", "predicted_risk",
                             "p_high", "p_medium", "p_low", "borderline", "G3", "actual_risk"]
                 if c in ranked.columns]
    st.dataframe(ranked[show_cols])

    st.download_button("Download this table as CSV",
                       ranked[show_cols].to_csv(index=False).encode("utf-8"),
                       file_name="spark_scored_students.csv", mime="text/csv")

    # Suggestions for one student chosen from the filtered table
    st.subheader("Suggestions for one student")

    def describe(i):
        row = ranked.loc[i]
        g2_text = "none" if pd.isna(row.get("G2", np.nan)) else int(row["G2"])
        return f"Row {i}, G1 {int(row['G1'])}, G2 {g2_text}, {row['predicted_risk']}"

    if ranked.empty:
        st.write("No students match the current filters.")
    else:
        pick = st.selectbox("Choose a student from the table above", ranked.index,
                            format_func=describe)
        chosen = ranked.loc[pick]
        chosen_g2 = chosen["G2"] if "G2" in ranked.columns else np.nan

        left, right = st.columns([1, 2])
        with left:
            show_risk(chosen["predicted_risk"])
            st.write("Stage used: " + ("Stage 2, G1 and G2" if chosen["stage"] == "stage2"
                                       else "Stage 1, G1 only"))
            st.write(f"Probability of High: {chosen['p_high']:.3f}")
            if chosen["borderline"] == "Yes":
                st.info("Borderline grade, treat the label as uncertain.")
        with right:
            steps = suggest_actions(chosen["predicted_risk"], chosen["stage"],
                                    chosen["G1"], chosen_g2, chosen["borderline"] == "Yes")
            if steps:
                st.markdown("\n".join("- " + step for step in steps))
                st.caption("General guidance based on the risk label and the grades, not a "
                           "finding of the model and not tested for effectiveness.")
            else:
                st.write("No action suggested for Low risk students.")
    
    # Check against final grades, only possible when G3 is in the file
    if "G3" in result.columns:
        with st.expander("Check against final grades, all students in the file"):
            table = (pd.crosstab(result["actual_risk"], result["predicted_risk"])
                     .reindex(index=RISK_ORDER, columns=RISK_ORDER, fill_value=0))
            st.dataframe(table)
            total = int((result["actual_risk"] == "High").sum())
            caught = int(((result["actual_risk"] == "High") & (result["predicted_risk"] == "High")).sum())
            st.write(f"High risk students caught: {caught} of {total}")


# ===== Tab 3, evidence =====
def evidence_tab():
    st.subheader("Saved results")
    files = {
        "Mathematics hold out": "v2_holdout_results.csv",
        "Portuguese cross subject test": "v2_portuguese_results.csv",
        "Six models, full metrics": "v2_six_models_full_metrics.csv",
        "Cross validation on fresh folds": "v2_fresh_folds.csv",
        "Rule baselines on development data": "v2_baselines.csv",
    }
    for title, name in files.items():
        with st.expander(title):
            path = TABLES_DIR / name
            if path.exists():
                st.dataframe(get_table(path))
            else:
                st.write(f"File not found, expected {name} in outputs/tables")

    st.subheader("How to read these results")
    st.markdown(
        "The first two grades carry almost all of the useful information. A simple rule that "
        "applies the risk bands to the second grade performs almost as well as the trained models.\n\n"
        "The test sets are small, 26 failing students in the Mathematics hold out and 70 in the "
        "Portuguese file, so differences of a few points between models are not reliable.\n\n"
        "The High threshold was chosen on development data for a recall of 0.90. On Portuguese "
        "students recall holds up but more flagged students turn out not to be at risk.\n\n"
        "The models were trained on secondary school Mathematics grades and have not been checked "
        "on university courses."
    )


# ===== Page layout =====
models = available_models()
if not models:
    st.error("No saved model bundles found in the models folder. Run notebook 12 first.")
    st.stop()

names = list(models)
choice = st.sidebar.selectbox("Model", names,
                              index=names.index(MAIN_MODEL) if MAIN_MODEL in names else 0)
bundle = get_bundle(models[choice])

st.sidebar.write("High thresholds in this model")
for stage_name, value in bundle["thresholds"].items():
    st.sidebar.write(f"{stage_name}: {value:.3f}")
st.sidebar.caption("The Random Forest is the main model. SVM at Stage 1 gave unreliable "
                   "probabilities, so avoid it in demonstrations.")

st.title("SPARK, student risk early warning")
st.caption("Decision support only. Risk bands from the final grade, High is 0 to 9, "
           "Medium is 10 to 13, and Low is 14 to 20.")

tab_single, tab_cohort, tab_evidence = st.tabs(["Single student", "Cohort", "Evidence"])
with tab_single:
    single_tab(bundle)
with tab_cohort:
    cohort_tab(bundle)
with tab_evidence:
    evidence_tab()