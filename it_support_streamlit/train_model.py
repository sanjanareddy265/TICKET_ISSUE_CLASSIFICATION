import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data" / "IT_Support_Tickets_Cleaned.csv")
text_col = "model_text" if "model_text" in df.columns else "clean_text"
X = df[text_col].fillna("").astype(str)
y = df["category"].astype(str)
valid = X.str.strip().ne("") & df["category"].notna()
X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=X))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

def vectorizer():
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42), cv=3),
}
rows, details = [], {}
for name, estimator in models.items():
    pipe = Pipeline([("tfidf", vectorizer()), ("classifier", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
        "Macro Precision": round(precision_score(y_test, pred, average="macro", zero_division=0), 4),
        "Macro Recall": round(recall_score(y_test, pred, average="macro", zero_division=0), 4),
        "Macro F1": round(f1_score(y_test, pred, average="macro", zero_division=0), 4),
        "Weighted F1": round(f1_score(y_test, pred, average="weighted", zero_division=0), 4),
    })
    details[name] = {
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique())).tolist(),
    }

comparison = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
best_name = comparison.loc[0, "Model"]
final_model = Pipeline([("tfidf", vectorizer()), ("classifier", models[best_name])])
final_model.fit(X, y)

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(final_model, ROOT / "models" / "ticket_classifier.joblib")
comparison.to_csv(ROOT / "models" / "model_comparison.csv", index=False)
with open(ROOT / "models" / "evaluation_details.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "split": "GroupShuffleSplit(test_size=0.20, random_state=42), grouping by identical model_text",
        "threshold": 0.70,
        "categories": sorted(y.unique()),
        "test_rows": int(len(test_idx)),
        "train_rows": int(len(train_idx)),
        "details": details,
    }, f, indent=2)
print("Best model:", best_name)
print(comparison.to_string(index=False))
