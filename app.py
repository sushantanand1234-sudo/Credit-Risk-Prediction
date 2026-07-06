from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import xgboost as xgb
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for Flask
import matplotlib.pyplot as plt

from werkzeug.utils import secure_filename
from sklearn.base import BaseEstimator, TransformerMixin

# ===================================================
# Education Mapping
# ===================================================

EDUCATION_MAP = {
    "SSC": 1,
    "OTHERS": 1,
    "12TH": 2,
    "GRADUATE": 3,
    "UNDER GRADUATE": 3,
    "PROFESSIONAL": 3,
    "POST-GRADUATE": 4
}

# ===================================================
# Custom Transformer
# ===================================================

class EducationOrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, mapping=EDUCATION_MAP, column="EDUCATION"):
        self.mapping = mapping
        self.column = column

    def fit(self, X, y=None):
        self.fallback_value_ = pd.Series(
            X[self.column].map(self.mapping)
        ).mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = (
            X[self.column]
            .map(self.mapping)
            .fillna(self.fallback_value_)
            .astype(int)
        )
        return X


# ===================================================
# Flask App
# ===================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
STATIC_FOLDER = "static"
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load models once at startup
education_encoder = joblib.load("models/education_encoder.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

model = xgb.XGBClassifier()
model.load_model("models/xgb_model.json")

label_encoder = joblib.load("models/label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded.", 400

    file = request.files["file"]

    if file.filename == "":
        return "No file selected.", 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return "Only CSV or Excel files allowed.", 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        if ext == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        X = education_encoder.transform(df)
        X = preprocessor.transform(X)

        prediction = model.predict(X)
        prediction = label_encoder.inverse_transform(prediction.astype(int))

        df["Prediction"] = prediction

        # ==========================================
        # Prediction Summary
        # ==========================================
        summary = df["Prediction"].value_counts()

        p1 = int(summary.get("P1", 0))
        p2 = int(summary.get("P2", 0))
        p3 = int(summary.get("P3", 0))
        p4 = int(summary.get("P4", 0))

        total = len(df)

        labels = ["P1", "P2", "P3", "P4"]
        counts = [p1, p2, p3, p4]

        # ==========================================
        # Pie Chart
        # ==========================================
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=labels, autopct="%1.1f%%")
        plt.title("Prediction Distribution")
        plt.savefig(os.path.join(STATIC_FOLDER, "pie_chart.png"), bbox_inches="tight")
        plt.close()

        # ==========================================
        # Bar Chart
        # ==========================================
        plt.figure(figsize=(6, 4))
        plt.bar(labels, counts)
        plt.title("Prediction Count")
        plt.xlabel("Classes")
        plt.ylabel("Customers")
        plt.savefig(os.path.join(STATIC_FOLDER, "bar_chart.png"), bbox_inches="tight")
        plt.close()

        # ==========================================
        # Save Output
        # ==========================================
        output_path = os.path.join(OUTPUT_FOLDER, "Predicted_Output.xlsx")
        df.to_excel(output_path, index=False)

        return render_template(
            "dashboard.html",
            total=total,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4
        )

    except Exception as e:
        return f"Error during prediction: {e}", 500


# ===========================================
# Download Route
# ===========================================

@app.route("/download")
def download():
    output_path = os.path.join(OUTPUT_FOLDER, "Predicted_Output.xlsx")

    if not os.path.exists(output_path):
        return "No prediction output available yet. Please run a prediction first.", 404

    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)