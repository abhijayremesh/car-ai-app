from flask import Flask, render_template_string, request
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

app = Flask(__name__)

# 1. Load data and train model using a Decision Tree (fully syllabus-compliant)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
dataset = read_csv(url, header=None)
data = dataset.values
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train = ordinal_encoder.fit_transform(X_train)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Using Decision Tree to handle non-linear rules natively
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)

# 2. HTML Frontend Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Value Predictor | NeuroPlan AI</title>
    <style>
        body { font-family: Arial, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .card { background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); width: 350px; }
        h2 { text-align: center; color: #38bdf8; margin-bottom: 20px; }
        label { display: block; margin-top: 10px; font-size: 14px; }
        select { width: 100%; padding: 8px; margin-top: 5px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: white; }
        button { width: 100%; background: #0ea5e9; color: white; border: none; padding: 10px; margin-top: 20px; border-radius: 6px; font-weight: bold; cursor: pointer; }
        button:hover { background: #0284c7; }
        .result { margin-top: 20px; text-align: center; font-size: 18px; font-weight: bold; color: #4ade80; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Car Value Predictor</h2>
        <form method="POST">
            <label>Buying Price:</label>
            <select name="buying"><option>v-high</option><option>high</option><option>med</option><option>low</option></select>
            
            <label>Maintenance Cost:</label>
            <select name="maint"><option>v-high</option><option>high</option><option>med</option><option>low</option></select>
            
            <label>Doors:</label>
            <select name="doors"><option>2</option><option>3</option><option>4</option><option>5-more</option></select>
            
            <label>Passenger Capacity:</label>
            <select name="persons"><option>2</option><option>4</option><option>more</option></select>
            
            <label>Luggage Boot Size:</label>
            <select name="lug_boot"><option>small</option><option>med</option><option>big</option></select>
            
            <label>Safety Rating:</label>
            <select name="safety"><option>low</option><option>med</option><option>high</option></select>
            
            <button type="submit">Predict Value Tier</button>
        </form>
        {% if prediction %}
            <div class="result">Prediction: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        user_car = [[
            request.form.get("buying"),
            request.form.get("maint"),
            request.form.get("doors"),
            request.form.get("persons"),
            request.form.get("lug_boot"),
            request.form.get("safety")
        ]]
        
        encoded_input = ordinal_encoder.transform(user_car)
        numeric_pred = model.predict(encoded_input)
        prediction = label_encoder.inverse_transform(numeric_pred)[0].upper()
        
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
