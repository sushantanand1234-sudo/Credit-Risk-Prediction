from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import xgboost as xgb
import os

from sklearn.base import BaseEstimator, TransformerMixin

# ===================================================
# Education Mapping
# ===================================================

EDUCATION_MAP = {
    "SSC":1,
    "OTHERS":1,
    "12TH":2,
    "GRADUATE":3,
    "UNDER GRADUATE":3,
    "PROFESSIONAL":3,
    "POST-GRADUATE":4
}

# ===================================================
# Custom Transformer
# ===================================================

class EducationOrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self,
                 mapping=EDUCATION_MAP,
                 column="EDUCATION"):

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

education_encoder = joblib.load("education_encoder.pkl")
preprocessor = joblib.load("preprocessor.pkl")

model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

label_encoder = joblib.load("label_encoder.pkl")


UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]

    if file.filename == "":
        return "No file selected."

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(filepath)

    if file.filename.endswith(".csv"):
        df = pd.read_csv(filepath)

    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(filepath)

    else:
        return "Only CSV or Excel files allowed."

    try:

        X = education_encoder.transform(df)

        X = preprocessor.transform(X)

        prediction = model.predict(X)

        prediction = label_encoder.inverse_transform(
            prediction.astype(int)
        )

        df["Prediction"] = prediction

        output_path = os.path.join(
            OUTPUT_FOLDER,
            "Predicted_Output.xlsx"
        )

        df.to_excel(output_path, index=False)

        return send_file(
            output_path,
            as_attachment=True
        )

    except Exception as e:

        return str(e)


if __name__ == "__main__":
    app.run(debug=True)