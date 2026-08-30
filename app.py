from flask import Flask, render_template_string, request
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

df = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header=None).values
X = df[:, :-1].astype(str)
y = df[:, -1].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_enc = oe.fit_transform(X_train)
X_test_enc = oe.transform(X_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_train_enc, y_train_enc)

acc = round(accuracy_score(y_test_enc, model.predict(X_test_enc)) * 100, 2)

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Torquex</title>
    <style>
        body { font-family: Arial, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .card { background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); width: 380px; }
        h2 { text-align: center; color: #38bdf8; margin-bottom: 10px; }
        .metrics { font-size: 12px; color: #94a3b8; text-align: center; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 10px; }
        label { display: block; margin-top: 8px; font-size: 13px; }
        select { width: 100%; padding: 6px; margin-top: 4px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: white; }
        button { width: 100%; background: #0ea5e9; color: white; border: none; padding: 10px; margin-top: 15px; border-radius: 6px; font-weight: bold; cursor: pointer; }
        button:hover { background: #0284c7; }
        .result { margin-top: 15px; text-align: center; font-size: 16px; font-weight: bold; color: #4ade80; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Torquex</h2>
        <div class="metrics">Model Accuracy: {{ acc }}%</div>
        <form method="POST">
            <label>Buying Price:</label>
            <select name="b"><option>v-high</option><option>high</option><option>med</option><option>low</option></select>
            <label>Maintenance Cost:</label>
            <select name="m"><option>v-high</option><option>high</option><option>med</option><option>low</option></select>
            <label>Doors:</label>
            <select name="d"><option>2</option><option>3</option><option>4</option><option>5-more</option></select>
            <label>Passenger Capacity:</label>
            <select name="p"><option>2</option><option>4</option><option>more</option></select>
            <label>Luggage Boot Size:</label>
            <select name="l"><option>small</option><option>med</option><option>big</option></select>
            <label>Safety Rating:</label>
            <select name="s"><option>low</option><option>med</option><option>high</option></select>
            <button type="submit">Predict</button>
        </form>
        {% if res %}
            <div class="result">Prediction: {{ res }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    res = None
    if request.method == "POST":
        inp = [[request.form.get("b"), request.form.get("m"), request.form.get("d"), request.form.get("p"), request.form.get("l"), request.form.get("s")]]
        res = le.inverse_transform(model.predict(oe.transform(inp)))[0].upper()
    return render_template_string(html, res=res, acc=acc)

if __name__ == "__main__":
    app.run(debug=True)
