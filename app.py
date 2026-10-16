from flask import Flask, render_template, request
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the existing car dataset from the repository.
df = read_csv("car.data", header=None).values

# The car.data file has six input columns and one target column.
X = df[:, :-1].astype(str)
y = df[:, -1].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=1,
    stratify=y,
)

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_train_encoded, y_train_encoded)

accuracy = round(
    accuracy_score(
        y_test_encoded,
        model.predict(X_test_encoded),
    ) * 100,
    2,
)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        values = [
            request.form.get("b", ""),
            request.form.get("m", ""),
            request.form.get("d", ""),
            request.form.get("p", ""),
            request.form.get("l", ""),
            request.form.get("s", ""),
        ]

        if all(values):
            try:
                encoded_values = encoder.transform([values])
                prediction = model.predict(encoded_values)[0]
                result = label_encoder.inverse_transform([prediction])[0].upper()
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
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
