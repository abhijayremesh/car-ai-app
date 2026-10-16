from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = "torquex_class12_secret"

# --- 1. MACHINE LEARNING MODEL PIPELINE ---

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

try:
    df = pd.read_csv(url, header=None).values
except Exception:
    df = pd.read_csv("car.data", header=None).values

X = df[:, :-1].astype(str)
y = df[:, -1].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

oe = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_enc = oe.fit_transform(X_train)
X_test_enc = oe.transform(X_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_enc, y_train_enc)

acc = round(
    accuracy_score(
        y_test_enc,
        model.predict(X_test_enc)
    ) * 100,
    2
)

feature_names = [
    "Buying Price",
    "Maintenance Cost",
    "Doors",
    "Passenger Capacity",
    "Luggage Boot",
    "Safety Rating"
]

importances = model.feature_importances_

feature_imp_dict = {
    feature_names[i]: round(float(importances[i]) * 100, 1)
    for i in range(len(feature_names))
}


# --- 2. AESTHETIC INTERACTIVE FRONTEND INTERFACE ---

html_template = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Torquex. - AI Vehicle Evaluation</title>

    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap"
          rel="stylesheet">
