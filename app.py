from flask import Flask, render_template, request
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ---------- Valuation settings (tweak these) ----------
CURRENCY = "AED"
BASE_PRICE = {"low": 40000, "med": 80000, "high": 150000, "vhigh": 250000}  # typical original price per tier
CLASS_FACTOR = {"unacc": 0.55, "acc": 0.75, "good": 0.90, "vgood": 1.00}
MAINT_FACTOR = {"low": 1.05, "med": 1.00, "high": 0.93, "vhigh": 0.85}
SAFETY_FACTOR = {"low": 0.85, "med": 1.00, "high": 1.08}
PERSONS_FACTOR = {"2": 0.90, "4": 1.00, "more": 1.03}
BOOT_FACTOR = {"small": 0.97, "med": 1.00, "big": 1.03}
AGE_FACTOR = {"under1": 0.95, "1to3": 0.82, "3to5": 0.65, "5to8": 0.48, "8to12": 0.33, "12plus": 0.20}
AGE_LABEL = {"under1": "Under 1 year", "1to3": "1–3 years", "3to5": "3–5 years",
             "5to8": "5–8 years", "8to12": "8–12 years", "12plus": "12+ years"}
RANGE = 0.12  # +/- 12% around the estimate

# ---------- Model ----------
df = read_csv("car.data", header=None).values
X = df[:, :-1]
y = df[:, -1].astype(str)

feature_names = [
    "Buying price", "Maintenance cost", "Number of doors",
    "Passenger capacity", "Luggage boot size", "Safety rating",
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

accuracy = round(accuracy_score(y_test_encoded, model.predict(X_test_encoded)) * 100, 2)

feature_importance = [
    {"name": name, "value": round(float(value) * 100, 1)}
    for name, value in zip(feature_names, model.feature_importances_)
]
feature_importance.sort(key=lambda item: item["value"], reverse=True)

RESULT_TEXT = {
    "UNACC": "Unacceptable: it fails at least one fundamental requirement.",
    "ACC": "Acceptable: it does the job, but has weaknesses holding it back.",
    "GOOD": "Good: a well-balanced car with few weak points.",
    "VGOOD": "Very good: strong on cost, practicality and safety.",
}


def predict_car(values):
    encoded_values = encoder.transform([values])
    prediction = model.predict(encoded_values)[0]
    return label_encoder.inverse_transform([prediction])[0].upper()


def explain(values, result):
    """Plain-English reasons, based on the rules in the car dataset."""
    b, m, d, p, l, s = values
    reasons = []

    # Deal-breakers
    if s == "low":
        reasons.append(("bad", "Low safety rating",
                        "This is a deal-breaker. In the dataset, every car with a low safety rating is unacceptable, whatever else it offers."))
    elif s == "high":
        reasons.append(("good", "High safety rating", "Top safety is the biggest single boost to a car's rating."))
    else:
        reasons.append(("warn", "Medium safety rating", "Okay, but a high rating is needed to reach the 'good' and 'very good' tiers."))

    if p == "2":
        reasons.append(("bad", "Only 2 passengers",
                        "This is also a deal-breaker: 2-seaters are rated unacceptable in this evaluation, however good they are otherwise."))
    elif p == "more":
        reasons.append(("good", "Seats more than 4", "High passenger capacity improves practicality."))
    else:
        reasons.append(("good", "Seats 4", "A practical, standard capacity."))

    # Cost
    if (b, m) in {("vhigh", "vhigh"), ("vhigh", "high"), ("high", "vhigh")}:
        reasons.append(("bad", "Buying + running cost is too high",
                        "A very high purchase price combined with high maintenance (or the reverse) makes the car unacceptable on cost alone."))
    else:
        if b in ("vhigh", "high"):
            reasons.append(("warn", "Expensive to buy", "A high purchase price caps how well the car can rate overall."))
        elif b == "low":
            reasons.append(("good", "Low purchase price", "Cheap to buy, which helps the rating."))
        if m in ("vhigh", "high"):
            reasons.append(("warn", "Costly to maintain", "High upkeep costs hold back the rating, even on a good car."))
        elif m == "low":
            reasons.append(("good", "Low maintenance cost", "Cheap to run, a strong plus for the rating."))

    # Boot
    if l == "small":
        reasons.append(("warn", "Small boot", "A small luggage space can keep an otherwise good car from reaching the top tier."))
    elif l == "big":
        reasons.append(("good", "Large boot", "More luggage space improves practicality."))

    # Doors
    reasons.append(("neutral", f"{d.replace('5more', '5+')} doors", "Door count has very little effect on the rating."))

    return {"summary": RESULT_TEXT.get(result, ""), "items": reasons}


def estimate_value(values, result, age):
    b, m, d, p, l, s = values
    base = BASE_PRICE[b]
    steps = [
        ("Car age (" + AGE_LABEL[age] + ")", AGE_FACTOR[age]),
        ("Overall condition rating (" + result.title() + ")", CLASS_FACTOR[result.lower()]),
        ("Maintenance cost (" + m + ")", MAINT_FACTOR[m]),
        ("Safety rating (" + s + ")", SAFETY_FACTOR[s]),
        ("Passenger capacity (" + p + ")", PERSONS_FACTOR[p]),
        ("Boot size (" + l + ")", BOOT_FACTOR[l]),
    ]
    total = base
    for _, factor in steps:
        total *= factor

    def money(v):
        return f"{CURRENCY} {round(v / 500) * 500:,.0f}"

    return {
        "mid": money(total),
        "low": money(total * (1 - RANGE)),
        "high": money(total * (1 + RANGE)),
        "base": money(base),
        "steps": [
            {"label": label, "pct": f"{(f - 1) * 100:+.0f}%", "good": f >= 1}
            for label, f in steps
        ],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_values = None
    upgrade_path = []
    reasons = None
    value = None

    if request.method == "POST":
        submitted_values = [
            request.form.get("b", ""), request.form.get("m", ""),
            request.form.get("d", ""), request.form.get("p", ""),
            request.form.get("l", ""), request.form.get("s", ""),
        ]
        age = request.form.get("a", "")

        if all(submitted_values) and age in AGE_FACTOR:
            try:
                result = predict_car(submitted_values)
                reasons = explain(submitted_values, result)
                value = estimate_value(submitted_values, result, age)

                upgrade_options = {
                    0: ["low", "med", "high", "vhigh"],
                    1: ["low", "med", "high", "vhigh"],
                    2: ["2", "3", "4", "5more"],
                    3: ["2", "4", "more"],
                    4: ["small", "med", "big"],
                    5: ["low", "med", "high"],
                }

                for idx, options in upgrade_options.items():
                    current_position = options.index(submitted_values[idx])
                    if current_position < len(options) - 1:
                        changed_values = submitted_values.copy()
                        changed_values[idx] = options[current_position + 1]
                        changed_result = predict_car(changed_values)

                        if changed_result != result:
                            upgrade_path.append({
                                "feature": feature_names[idx],
                                "from_value": submitted_values[idx],
                                "to_value": options[current_position + 1],
                                "new_result": changed_result,
                            })

                upgrade_path.sort(
                    key=lambda item: next(
                        f["value"] for f in feature_importance if f["name"] == item["feature"]
                    ),
                    reverse=True,
                )
            except Exception:
                error = "Please complete every vehicle detail before evaluating."
        else:
            error = "Please complete every vehicle detail before evaluating."

    return render_template(
        "index.html",
        result=result, res=result, error=error,
        accuracy=accuracy, acc=accuracy,
        feature_importance=feature_importance,
        upgrade_path=upgrade_path,
        submitted_values=submitted_values,
        reasons=reasons, value=value,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
