from flask import Flask, render_template_string, request, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import io

app = Flask(__name__)
app.secret_key = "torquex_class12_secret"

# --- 1. MACHINE LEARNING MODEL PIPELINE ---
# Load the classic Car Evaluation dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
try:
    df = pd.read_csv(url, header=None).values
except Exception:
    df = pd.read_csv("car.data", header=None).values

X = df[:, :-1].astype(str)
y = df[:, -1].astype(str)

# Split dataset into Training (67%) and Testing (33%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Encode categorical input features into numerical values
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_enc = oe.fit_transform(X_train)
X_test_enc = oe.transform(X_test)

# Encode target labels (unacc, acc, good, vgood)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Train the Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_enc, y_train_enc)

# Calculate model accuracy for display
acc = round(accuracy_score(y_test_enc, model.predict(X_test_enc)) * 100, 2)

# Extract feature importances to show how the AI makes decisions
feature_names = ["Buying Price", "Maintenance Cost", "Doors", "Passenger Capacity", "Luggage Boot", "Safety Rating"]
importances = model.feature_importances_
feature_imp_dict = {feature_names[i]: round(float(importances[i]) * 100, 1) for i in range(len(feature_names))}


# --- 2. HTML & CSS FRONTEND INTERFACE ---
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Torquex - AI Car Evaluator</title>
    <style>
        :root {
            --bg: #090d16;
            --card-bg: #131c2e;
            --border: #1e293b;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #4ade80;
        }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { width: 100%; max-width: 720px; background: var(--card-bg); padding: 30px; border-radius: 16px; box-shadow: 0 20px 40px rgba(0,0,0,0.6); border: 1px solid var(--border); }
        h2 { text-align: center; color: var(--accent); margin-bottom: 5px; font-size: 28px; }
        .subtitle { text-align: center; color: var(--text-muted); font-size: 13px; margin-bottom: 20px; }
        .metrics-bar { display: flex; justify-content: space-around; background: #0f172a; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-size: 13px; border: 1px solid var(--border); }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
        .tab-btn { background: transparent; border: none; color: var(--text-muted); padding: 8px 16px; cursor: pointer; font-weight: bold; border-radius: 6px; font-size: 14px; }
        .tab-btn.active { background: var(--accent); color: #090d16; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { margin-bottom: 12px; }
        label { display: block; margin-bottom: 5px; font-size: 13px; color: var(--text-muted); font-weight: 500; }
        select, input[type="file"] { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid var(--border); background: #0b0f19; color: white; font-size: 14px; outline: none; box-sizing: border-box; }
        select:focus, input:focus { border-color: var(--accent); }
        .btn { width: 100%; background: var(--accent); color: #090d16; border: none; padding: 12px; margin-top: 15px; border-radius: 8px; font-weight: bold; font-size: 15px; cursor: pointer; transition: background 0.2s; }
        .btn:hover { background: var(--accent-hover); }
        .result-box { margin-top: 20px; padding: 15px; text-align: center; font-size: 18px; font-weight: bold; background: #0f172a; border-radius: 8px; border: 1px solid var(--border); color: var(--success); }
        .feature-imp { margin-top: 25px; padding-top: 15px; border-top: 1px solid var(--border); }
        .feature-imp h4 { margin: 0 0 10px 0; font-size: 14px; color: var(--text-muted); }
        .imp-bar-container { margin-bottom: 8px; font-size: 12px; }
        .imp-bar-bg { background: #0b0f19; border-radius: 4px; height: 8px; overflow: hidden; margin-top: 3px; border: 1px solid var(--border); }
        .imp-bar-fill { background: var(--accent); height: 100%; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }
        th, td { padding: 8px; border: 1px solid var(--border); text-align: center; }
        th { background: #0f172a; color: var(--accent); }
    </style>
</head>
<body>
    <div class="container">
        <h2>Torquex</h2>
        <div class="subtitle">AI-Powered Vehicle Evaluation System (Class 12 Board Project)</div>
        
        <div class="metrics-bar">
            <span>Algorithm: <b>Decision Tree</b></span>
            <span>Accuracy: <b style="color: var(--success);">{{ acc }}%</b></span>
            <span>Dataset: <b>UCI Car Evaluation</b></span>
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('single', event)">Single Prediction</button>
            <button class="tab-btn" onclick="switchTab('batch', event)">Batch CSV Predictor</button>
        </div>

        <!-- Single Prediction Tab -->
        <div id="single" class="tab-content active">
            <form method="POST">
                <input type="hidden" name="form_type" value="single">
                <div class="grid">
                    <div class="form-group">
                        <label>Buying Price:</label>
                        <select name="b"><option>v-high</option><option>high</option><option>med</option><option>low</option></select>
                    </div>
                    <div class="form-group">
                        <label>Maintenance Cost:</label>
                        <select name="m"><option>v-high</option><option>high</option><option>med</option><option>low</option></select>
                    </div>
                    <div class="form-group">
                        <label>Doors:</label>
                        <select name="d"><option>2</option><option>3</option><option>4</option><option>5-more</option></select>
                    </div>
                    <div class="form-group">
                        <label>Passenger Capacity:</label>
                        <select name="p"><option>2</option><option>4</option><option>more</option></select>
                    </div>
                    <div class="form-group">
                        <label>Luggage Boot Size:</label>
                        <select name="l"><option>small</option><option>med</option><option>big</option></select>
                    </div>
                    <div class="form-group">
                        <label>Safety Rating:</label>
                        <select name="s"><option>low</option><option>med</option><option>high</option></select>
                    </div>
                </div>
                <button type="submit" class="btn">Evaluate Vehicle</button>
            </form>

            {% if res %}
                <div class="result-box">Prediction Result: {{ res }}</div>
            {% endif %}

            <!-- Feature Importance Analysis -->
            <div class="feature-imp">
                <h4>AI Feature Importance Weights</h4>
                {% for feat, score in importances.items() %}
                <div class="imp-bar-container">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{{ feat }}</span>
                        <span style="color: var(--text-muted);">{{ score }}%</span>
                    </div>
                    <div class="imp-bar-bg">
                        <div class="imp-bar-fill" style="width: {{ score }}%;"></div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Batch CSV Tab -->
        <div id="batch" class="tab-content">
            <form method="POST" enctype="multipart/form-data">
                <input type="hidden" name="form_type" value="batch">
                <label>Upload CSV File (6 comma-separated features per row):</label>
                <input type="file" name="file" accept=".csv" required style="margin-top: 8px;">
                <button type="submit" class="btn">Process Batch CSV</button>
            </form>
            <div style="text-align: center; margin-top: 12px;">
                <a href="/download-sample" style="color: var(--accent); font-size: 13px; text-decoration: none;">Download Sample CSV Template</a>
            </div>

            {% if batch_results %}
            <div style="margin-top: 20px; max-height: 240px; overflow-y: auto;">
                <table>
                    <tr>
                        <th>Buying</th><th>Maint</th><th>Doors</th><th>Persons</th><th>Boot</th><th>Safety</th><th>Prediction</th>
                    </tr>
                    {% for row in batch_results %}
                    <tr>
                        <td>{{ row.buying }}</td>
                        <td>{{ row.maint }}</td>
                        <td>{{ row.doors }}</td>
                        <td>{{ row.persons }}</td>
                        <td>{{ row.lug_boot }}</td>
                        <td>{{ row.safety }}</td>
                        <td style="color: var(--success); font-weight: bold;">{{ row.prediction }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        function switchTab(tabId, evt) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            evt.currentTarget.classList.add('active');
        }
    </script>
</body>
</html>
"""


# --- 3. FLASK ROUTING & REQUEST HANDLING ---
@app.route("/", methods=["GET", "POST"])
def index():
    res = None
    batch_results = None
    if request.method == "POST":
        form_type = request.form.get("form_type")
        
        # Handle Single User Form Input
        if form_type == "single":
            inp = [[
                request.form.get("b"),
                request.form.get("m"),
                request.form.get("d"),
                request.form.get("p"),
                request.form.get("l"),
                request.form.get("s")
            ]]
            res = le.inverse_transform(model.predict(oe.transform(inp)))[0].upper()
            
        # Handle Batch CSV Upload Input
        elif form_type == "batch":
            file = request.files.get("file")
            if file:
                df_batch = pd.read_csv(file, header=None)
                X_batch = df_batch.iloc[:, :6].astype(str).values
                preds = le.inverse_transform(model.predict(oe.transform(X_batch)))
                
                batch_results = []
                for i, row in enumerate(X_batch):
                    batch_results.append({
                        "buying": row[0], "maint": row[1], "doors": row[2],
                        "persons": row[3], "lug_boot": row[4], "safety": row[5],
                        "prediction": preds[i].upper()
                    })

    return render_template_string(html_template, acc=acc, res=res, importances=feature_imp_dict, batch_results=batch_results)

@app.route("/download-sample")
def download_sample():
    sample_csv = "v-high,high,2,2,small,low\nmed,med,4,4,med,med\nlow,low,5-more,more,big,high"
    return send_file(
        io.BytesIO(sample_csv.encode('utf-8')),
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_car_data.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)
