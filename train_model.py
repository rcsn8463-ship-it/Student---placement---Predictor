
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_data.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

features = ["CGPA","Attendance","Internships","Projects","DSA_Score","Communication_Score","Certifications","Aptitude_Score"]
df = pd.read_csv(DATA)

X = df[features]
y = df["Placement"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    {"n_estimators":[150,250], "max_depth":[None,8,12], "min_samples_split":[2,5]},
    cv=5, scoring="f1", n_jobs=-1
)
clf_grid.fit(X_train, y_train)
clf = clf_grid.best_estimator_

pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:,1]
metrics = {
    "accuracy": accuracy_score(y_test,pred),
    "precision": precision_score(y_test,pred,zero_division=0),
    "recall": recall_score(y_test,pred,zero_division=0),
    "f1": f1_score(y_test,pred,zero_division=0),
    "roc_auc": roc_auc_score(y_test,prob)
}

placed = df[df["Placement"]==1].dropna(subset=["Package_LPA"])
Xr = placed[features]
yr = placed["Package_LPA"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
reg.fit(Xr_train, yr_train)
rpred = reg.predict(Xr_test)
reg_metrics = {"mae": mean_absolute_error(yr_test,rpred), "r2": r2_score(yr_test,rpred)}

joblib.dump(clf, MODELS/"placement_model.pkl")
joblib.dump(reg, MODELS/"package_model.pkl")
joblib.dump(features, MODELS/"features.pkl")
joblib.dump({"classification":metrics,"regression":reg_metrics}, MODELS/"metrics.pkl")
print("Models trained successfully.")
print(metrics)
print(reg_metrics)
