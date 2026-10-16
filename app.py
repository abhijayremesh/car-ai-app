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
    accuracy_score(y_test_enc, model.predict(X_test_enc)) * 100,
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

    <style>
        :root {
            --bg-color: #060911;
            --card-bg: rgba(13, 20, 38, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.25);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #4ade80;
        }

        * {
            box-sizing: border-box;
            transition: all 0.25s ease;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);

            background-image:
                radial-gradient(
                    circle at 15% 10%,
                    rgba(56, 189, 248, 0.06) 0%,
                    transparent 40%
                ),
                radial-gradient(
                    circle at 85% 90%,
                    rgba(2, 132, 199, 0.04) 0%,
                    transparent 40%
                );

            color: var(--text-main);
            margin: 0;
            min-height: 100vh;

            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;

            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 760px;

            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);

            border: 1px solid var(--border-color);
            border-radius: 24px;

            padding: 35px;

            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6);

            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(15px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .header-section {
            text-align: center;
            margin-bottom: 25px;
        }

        .logo {
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 2px;
            color: var(--text-main);
            text-transform: lowercase;
            margin-bottom: 4px;
        }

        .logo span {
            color: var(--accent);
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        /* Metrics Bar */

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;

            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);

            padding: 14px;
            border-radius: 14px;
            margin-bottom: 25px;

            text-align: center;
        }

        .metric-item span:first-child {
            display: block;
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 3px;
        }

        .metric-item span:last-child {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-main);
        }

        /* Form Layout */

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }

        .input-group {
            background: rgba(10, 15, 30, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 12px 16px;
        }

        .input-group:hover {
            border-color: rgba(56, 189, 248, 0.4);
        }

        label {
            display: block;
            font-size: 11px;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 500;
        }

        select {
            width: 100%;
            background: transparent;
            border: none;
            color: var(--text-main);
            font-size: 14px;
            font-family: 'Outfit', sans-serif;
            font-weight: 500;
            outline: none;
            cursor: pointer;
        }

        select option {
            background: #0d1426;
            color: #f8fafc;
        }

        .submit-btn {
            width: 100%;
            padding: 14px;

            background: linear-gradient(
                135deg,
                #38bdf8 0%,
                #0284c7 100%
            );

            color: #060911;
            border: none;
            border-radius: 14px;

            font-size: 15px;
            font-weight: 700;

            cursor: pointer;

            box-shadow: 0 4px 15px var(--accent-glow);
            letter-spacing: 0.5px;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4);
        }

        /* Result Animation Box */

        .result-box {
            margin-top: 20px;

            background: rgba(74, 222, 128, 0.05);
            border: 1px solid rgba(74, 222,
