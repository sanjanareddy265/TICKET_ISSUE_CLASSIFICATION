# IT Support Ticket Intelligence

A Streamlit prototype for classifying IT support tickets, recommending a support queue, and exploring ticket analytics.

## Features
- Predicts one of seven ticket categories from a new issue description.
- Displays calibrated class probability and a routing recommendation.
- Sends low-confidence predictions to manual review (prototype threshold: 70%).
- Dashboard for category, priority, monthly trend, resolution time, and assignment-group analytics.
- Exact repeated-description analysis.
- Model comparison and confusion matrix.

## Model and evaluation
The training/evaluation process follows the supplied notebooks' TF-IDF setup (max_features=5000, ngram_range=(1,2), min_df=2) and compares Logistic Regression, Multinomial Naive Bayes, and calibrated Linear SVM.

For evaluation, this deployment package uses an 80/20 `GroupShuffleSplit` grouped by identical `model_text` (the duplicate-aware split approach in HCL3). Models are compared using Macro F1; the winning pipeline is refitted on all labelled rows for the live demo. See `models/model_comparison.csv` and `models/evaluation_details.json`.

The model artifact is already included at `models/ticket_classifier.joblib`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload the contents of this project folder.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy. Keep `data/` and `models/` in the repository because the app reads these files at startup.

## Dataset
- `data/IT_Support_Tickets_Cleaned.csv`: labelled dataset used to train the text classifier.
- `data/tickets_raw.csv`: raw dataset used for dashboard and recurring-description analysis.

## Important limitations
- The model is a hackathon decision-support prototype, not a production ITSM integration.
- The app recommends a queue; it does not route, close, or modify real tickets.
- The 70% threshold is a demo rule and should be validated before operational use.
- The recurring-issue view counts exact repeated text/category pairs, not semantically similar issues.
- Dataset category imbalance remains; review per-class metrics, especially for minority categories.
