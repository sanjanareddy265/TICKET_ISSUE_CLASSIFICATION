import re
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="IT Support Ticket Intelligence", page_icon="🎫", layout="wide")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "ticket_classifier.joblib"
CLEAN_PATH = ROOT / "data" / "IT_Support_Tickets_Cleaned.csv"
RAW_PATH = ROOT / "data" / "tickets_raw.csv"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"
DETAILS_PATH = ROOT / "models" / "evaluation_details.json"
THRESHOLD = 0.70

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def clean_new_ticket(text):
    """Same cleaning rules used in HCL1, applied to a newly entered ticket."""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data
def load_evaluation():
    with open(DETAILS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
clean_df = load_data(CLEAN_PATH)
raw_df = load_data(RAW_PATH)
comparison = load_data(COMPARISON_PATH)
evaluation = load_evaluation()

st.title("🎫 IT Support Ticket Intelligence")
st.caption("Classify incoming IT tickets, recommend the right support queue, and explore ticket trends.")
st.info("This is a decision-support prototype. It recommends a category and routing action; it does not change or close real ITSM tickets.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Classify a Ticket", "📊 Analytics Dashboard", "🔁 Recurring Issues", "🧪 Model Performance"
])

with tab1:
    st.subheader("Classify a new IT support ticket")
    ticket_text = st.text_area(
        "Describe the issue",
        placeholder="Example: I cannot connect to the office VPN and need access urgently...",
        height=150
    )
    if st.button("Predict Ticket Category", type="primary", use_container_width=False):
        cleaned = clean_new_ticket(ticket_text)
        if not cleaned:
            st.warning("Please enter a ticket description first.")
        else:
            predicted = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]
            classes = model.named_steps["classifier"].classes_
            confidence = float(np.max(probabilities))
            ranking = np.argsort(probabilities)[::-1][:3]
            status = "Auto Route Recommendation" if confidence >= THRESHOLD else "Manual Review Recommended"

            st.markdown("### Prediction")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Category", predicted)
            c2.metric("Confidence", f"{confidence*100:.1f}%")
            c3.metric("Routing Recommendation", status)
            if confidence >= THRESHOLD:
                st.success(f"Suggested queue: **{predicted}**. Confidence is at or above the {THRESHOLD:.0%} demo threshold.")
            else:
                st.warning(f"Send for human review. Confidence is below the {THRESHOLD:.0%} demo threshold.")
            st.caption("Confidence is the calibrated classifier probability for the predicted class. The 70% routing threshold is a prototype setting, not a production-approved policy.")
            prob_df = pd.DataFrame({
                "Category": [classes[i] for i in ranking],
                "Probability": [float(probabilities[i]) for i in ranking]
            })
            st.write("Top category probabilities")
            st.dataframe(
                prob_df.style.format({"Probability": "{:.1%}"}),
                use_container_width=True,
                hide_index=True
            )
            st.progress(min(max(confidence, 0.0), 1.0))

    st.divider()
    st.caption("Try a sample ticket")
    samples = [
        "Unable to connect to office VPN network",
        "Laptop screen is not turning on",
        "Outlook emails are not syncing",
        "Employee account password reset required",
        "Antivirus software showing security alert"
    ]
    sample = st.selectbox("Choose a sample", ["Select a sample..."] + samples)
    if sample != "Select a sample...":
        st.code(sample)

with tab2:
    st.subheader("Ticket overview")
    total = len(raw_df)
    categories = raw_df["category"].nunique() if "category" in raw_df else 0
    avg_resolution = pd.to_numeric(raw_df.get("resolution_hours"), errors="coerce").mean()
    reopened = raw_df["reopened"].astype(str).str.lower().isin(["true", "1", "yes"]).sum() if "reopened" in raw_df else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Tickets", f"{total:,}")
    k2.metric("Ticket Categories", categories)
    k3.metric("Avg. Resolution Time", f"{avg_resolution:.1f} hrs" if pd.notna(avg_resolution) else "N/A")
    k4.metric("Reopened Tickets", f"{reopened:,}")

    left, right = st.columns(2)
    with left:
        cat_counts = raw_df["category"].value_counts().rename_axis("Category").reset_index(name="Tickets")
        st.plotly_chart(px.bar(cat_counts, x="Category", y="Tickets", title="Tickets by Category", text="Tickets"), use_container_width=True)
    with right:
        if "priority" in raw_df:
            priority_counts = raw_df["priority"].fillna("Unknown").value_counts().rename_axis("Priority").reset_index(name="Tickets")
            st.plotly_chart(px.pie(priority_counts, names="Priority", values="Tickets", title="Priority Distribution", hole=0.4), use_container_width=True)

    if "created_at" in raw_df:
        trend = raw_df.copy()
        trend["created_at"] = pd.to_datetime(trend["created_at"], errors="coerce")
        trend = trend.dropna(subset=["created_at"])
        if not trend.empty:
            trend["Month"] = trend["created_at"].dt.to_period("M").astype(str)
            monthly = trend.groupby(["Month", "category"]).size().reset_index(name="Tickets")
            st.plotly_chart(px.line(monthly, x="Month", y="Tickets", color="category", markers=True, title="Monthly Ticket Trend"), use_container_width=True)

    if "assignment_group" in raw_df:
        groups = raw_df["assignment_group"].fillna("Unknown").value_counts().head(10).rename_axis("Assignment Group").reset_index(name="Tickets")
        st.plotly_chart(px.bar(groups, x="Tickets", y="Assignment Group", orientation="h", title="Top Assignment Groups"), use_container_width=True)

with tab3:
    st.subheader("Repeated ticket descriptions")
    st.caption("This view counts exact repeated descriptions. Similar-but-differently-worded issues are not grouped semantically.")
    source = raw_df.copy()
    if "short_description" in source and "description" in source:
        source["issue_text"] = (
            source["short_description"].fillna("").astype(str).str.strip() + " " +
            source["description"].fillna("").astype(str).str.strip()
        ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    else:
        source["issue_text"] = source.get("model_text", "").fillna("").astype(str).str.lower()
    repeated = (
        source[source["issue_text"].ne("")]
        .groupby(["issue_text", "category"], dropna=False)
        .size().reset_index(name="Frequency")
    )
    repeated = repeated[repeated["Frequency"] > 1].sort_values("Frequency", ascending=False)
    if repeated.empty:
        st.info("No exact repeated descriptions were found.")
    else:
        st.metric("Repeated description/category groups", len(repeated))
        st.dataframe(repeated.head(30).rename(columns={"issue_text": "Ticket Description"}), use_container_width=True, hide_index=True)
        top = repeated.head(10).copy()
        top["Ticket Description"] = top["issue_text"].str.slice(0, 75)
        st.plotly_chart(px.bar(top, x="Frequency", y="Ticket Description", color="category", orientation="h", title="Top Repeated Ticket Descriptions"), use_container_width=True)

with tab4:
    st.subheader("Model evaluation")
    st.caption("Evaluation uses a 80/20 GroupShuffleSplit grouped by identical model_text, matching the duplicate-aware approach in HCL3. The selected model is refitted on all labelled rows for deployment.")
    st.write(f"**Deployed model:** {evaluation['best_model']}")
    st.write(f"**Evaluation split:** {evaluation['split']}")
    st.write(f"**Training rows:** {evaluation['train_rows']:,}  |  **Test rows:** {evaluation['test_rows']:,}")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    details = evaluation["details"][evaluation["best_model"]]
    labels = evaluation["categories"]
    cm = np.array(details["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", title=f"Confusion Matrix — {evaluation['best_model']}", labels={"x":"Predicted category", "y":"Actual category", "color":"Tickets"}), use_container_width=True)

    report = pd.DataFrame(details["report"]).T
    if all(c in report.columns for c in ["precision", "recall", "f1-score"]) and set(labels).issubset(report.index):
        metric_df = report.loc[labels, ["precision", "recall", "f1-score"]].reset_index(names="Category")
        metric_long = metric_df.melt(id_vars="Category", var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(metric_long, x="Category", y="Score", color="Metric", barmode="group", title="Category-wise Precision, Recall and F1"), use_container_width=True)
        st.dataframe(report.loc[labels, ["precision", "recall", "f1-score", "support"]].style.format({"precision":"{:.2f}", "recall":"{:.2f}", "f1-score":"{:.2f}"}), use_container_width=True)

st.sidebar.title("About this project")
st.sidebar.write("**Use case:** IT Support Ticket Classification and Analytics")
st.sidebar.write("**Text features:** TF-IDF (up to 5,000 features; unigrams and bigrams)")
st.sidebar.write(f"**Model:** {evaluation['best_model']}")
st.sidebar.write(f"**Routing threshold:** {THRESHOLD:.0%}")
st.sidebar.caption("Prototype for hackathon demonstration. Validate on representative operational data before real-world use.")
