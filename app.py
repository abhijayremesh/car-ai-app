from flask import Flask, render_template, request
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

df = read_csv("car.data", header=None).values
X = df[:, :-1]
y = df[:, -1].astype(str)

feature_names = [
    "Buying price",
    "Maintenance cost",
    "Number of doors",
    "Passenger capacity",
    "Luggage boot size",
    "Safety rating",
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1, stratify=y
)

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_train_encoded, y_train_encoded)

accuracy = round(
    accuracy_score(y_test_encoded, model.predict(X_test_encoded)) * 100, 2
)

feature_importance = [
    {
        "name": name,
        "value": round(float(value) * 100, 1),
    }
    for name, value in zip(feature_names, model.feature_importances_)
]
feature_importance.sort(key=lambda item: item["value"], reverse=True)


def predict_car(values):
    encoded_values = encoder.transform([values])
    prediction = model.predict(encoded_values)[0]
    return label_encoder.inverse_transform([prediction])[0].upper()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_values = None
    upgrade_path = []

    if request.method == "POST":
        submitted_values = [
            request.form.get("b", ""),
            request.form.get("m", ""),
            request.form.get("d", ""),
            request.form.get("p", ""),
            request.form.get("l", ""),
            request.form.get("s", ""),
        ]

        if all(submitted_values):
            try:
                result = predict_car(submitted_values)

                upgrade_options = {
                    0: ["low", "med", "high", "vhigh"],
                    1: ["low", "med", "high", "vhigh"],
                    2: ["2", "3", "4", "5more"],
                    3: ["2", "4", "more"],
                    4: ["small", "med", "big"],
                    5: ["low", "med", "high"],
                }

                for index, options in upgrade_options.items():
                    current_position = options.index(submitted_values[index])
                    if current_position < len(options) - 1:
                        changed_values = submitted_values.copy()
                        changed_values[index] = options[current_position + 1]
                        changed_result = predict_car(changed_values)

                        if changed_result != result:
                            upgrade_path.append({
                                "feature": feature_names[index],
                                "from_value": submitted_values[index],
                                "to_value": options[current_position + 1],
                                "new_result": changed_result,
                            })

                upgrade_path.sort(
                    key=lambda item: next(
                        feature["value"]
                        for feature in feature_importance
                        if feature["name"] == item["feature"]
                    ),
                    reverse=True,
                )

            except Exception:
                error = "Please complete every vehicle detail before evaluating."
        else:
            error = "Please complete every vehicle detail before evaluating."

    return render_template(
        "index.html",
        result=result,
        res=result,
        error=error,
        accuracy=accuracy,
        acc=accuracy,
        feature_importance=feature_importance,
        upgrade_path=upgrade_path,
        submitted_values=submitted_values,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
