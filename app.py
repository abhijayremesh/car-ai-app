from flask import Flask, render_template_string, request
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and train the model without changing the original prediction logic.
df = read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    header=None,
).values

X = df[:, :-1].astype(str)
y = df[:, -1].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=1,
)

ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
X_train_encoded = ordinal_encoder.fit_transform(X_train)
X_test_encoded = ordinal_encoder.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_train_encoded, y_train_encoded)

# Keep the original model evaluation available internally without displaying it.
accuracy_score(y_test_encoded, model.predict(X_test_encoded))

html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Torquex | Car Finder</title>
    <style>
        :root {
            color-scheme: dark;
            --background: #07111f;
            --surface: rgba(15, 29, 49, 0.88);
            --surface-light: rgba(26, 48, 77, 0.75);
            --border: rgba(148, 163, 184, 0.2);
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-dark: #0284c7;
            --success: #4ade80;
        }

        * { box-sizing: border-box; }

        body {
            min-height: 100vh;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
            overflow-x: hidden;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 15% 10%, rgba(14, 165, 233, 0.2), transparent 34%),
                radial-gradient(circle at 85% 90%, rgba(37, 99, 235, 0.16), transparent 32%),
                var(--background);
        }

        .card {
            width: min(100%, 500px);
            padding: clamp(24px, 5vw, 42px);
            border: 1px solid var(--border);
            border-radius: 28px;
            background: var(--surface);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(18px);
            animation: rise 0.65s ease-out both;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }

        .brand-mark {
            display: grid;
            width: 42px;
            height: 42px;
            place-items: center;
            border-radius: 14px;
            color: #082f49;
            background: linear-gradient(135deg, #7dd3fc, #38bdf8);
            font-size: 22px;
            font-weight: 900;
            box-shadow: 0 8px 24px rgba(56, 189, 248, 0.28);
        }

        h1 {
            margin: 0;
            font-size: clamp(28px, 7vw, 40px);
            letter-spacing: -0.05em;
        }

        .intro {
            margin: 0 0 26px 54px;
            color: var(--muted);
            line-height: 1.55;
        }

        form { display: grid; gap: 17px; }

        .field { display: grid; gap: 8px; }

        label {
            color: #cbd5e1;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        select {
            width: 100%;
            min-height: 48px;
            padding: 0 14px;
            border: 1px solid var(--border);
            border-radius: 13px;
            outline: none;
            color: var(--text);
            background: var(--surface-light);
            font: inherit;
            cursor: pointer;
            transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
        }

        select:hover { border-color: rgba(56, 189, 248, 0.55); }
        select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.14);
        }

        button {
            min-height: 52px;
            margin-top: 6px;
            border: 0;
            border-radius: 14px;
            color: #082f49;
            background: linear-gradient(135deg, #7dd3fc, #38bdf8);
            font: inherit;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 12px 28px rgba(14, 165, 233, 0.25);
            transition: transform 0.2s, box-shadow 0.2s, filter 0.2s;
        }

        button:hover {
            transform: translateY(-2px);
            filter: brightness(1.08);
            box-shadow: 0 16px 32px rgba(14, 165, 233, 0.35);
        }

        button:active { transform: translateY(0); }

        .result {
            margin-top: 24px;
            padding: 18px;
            border: 1px solid rgba(74, 222, 128, 0.25);
            border-radius: 15px;
            color: var(--success);
            background: rgba(74, 222, 128, 0.08);
            text-align: center;
            font-weight: 800;
            animation: reveal 0.35s ease-out both;
        }

        @keyframes rise {
            from { opacity: 0; transform: translateY(18px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes reveal {
            from { opacity: 0; transform: scale(0.97); }
            to { opacity: 1; transform: scale(1); }
        }

        @media (max-width: 480px) {
            body { padding: 14px; }
            .card { border-radius: 22px; padding: 24px 18px; }
            .intro { margin-left: 0; }
        }
    </style>
</head>
<body>
    <main class="card">
        <div class="brand">
            <div class="brand-mark">T</div>
            <h1>Torquex</h1>
        </div>
        <p class="intro">Find the car that fits your priorities.</p>

        <form method="POST">
            <div class="field">
                <label for="b">Buying price</label>
                <select id="b" name="b">
                    <option value="v-high">Very high</option>
                    <option value="high">High</option>
                    <option value="med">Medium</option>
                    <option value="low">Low</option>
                </select>
            </div>

            <div class="field">
                <label for="m">Maintenance cost</label>
                <select id="m" name="m">
                    <option value="v-high">Very high</option>
                    <option value="high">High</option>
                    <option value="med">Medium</option>
                    <option value="low">Low</option>
                </select>
            </div>

            <div class="field">
                <label for="d">Doors</label>
                <select id="d" name="d">
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="5-more">5 or more</option>
                </select>
            </div>

            <div class="field">
                <label for="p">Passenger capacity</label>
                <select id="p" name="p">
                    <option value="2">2</option>
                    <option value="4">4</option>
                    <option value="more">More than 4</option>
                </select>
            </div>

            <div class="field">
                <label for="l">Luggage boot size</label>
                <select id="l" name="l">
                    <option value="small">Small</option>
                    <option value="med">Medium</option>
                    <option value="big">Large</option>
                </select>
            </div>

            <div class="field">
                <label for="s">Safety rating</label>
                <select id="s" name="s">
                    <option value="low">Low</option>
                    <option value="med">Medium</option>
                    <option value="high">High</option>
                </select>
            </div>

            <button type="submit">Find my car</button>
        </form>

        {% if res %}
            <div class="result" role="status">Recommended category: {{ res }}</div>
        {% endif %}
    </main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        inputs = [[
            request.form.get("b"),
            request.form.get("m"),
            request.form.get("d"),
            request.form.get("p"),
            request.form.get("l"),
            request.form.get("s"),
        ]]
        encoded_inputs = ordinal_encoder.transform(inputs)
        prediction = model.predict(encoded_inputs)
        result = label_encoder.inverse_transform(prediction)[0].upper()

    return render_template_string(html, res=result)


if __name__ == "__main__":
    app.run(debug=True)


