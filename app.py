from datetime import date

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

# ---------- Settings ----------
CURRENCY = "AED"
SAR_TO_AED = 3.6725 / 3.75      # both currencies are pegged to the USD
DATA_YEAR = 2021                # year the listings were collected
CURRENT_YEAR = date.today().year
MIN_YEAR = 2000

CATS = ["Make", "Type", "Gear_Type", "Options", "Fuel_Type"]
NUMS = ["Age", "Mileage", "Engine_Size"]
FEATURES = CATS + NUMS

# ---------- Data ----------
raw = pd.read_csv("UsedCarsSA_Clean_EN.csv").drop_duplicates()
# Price 0 = "negotiable" listing with no price; the rest removes obvious junk rows.
df = raw[
    (raw.Price >= 5000)
    & (raw.Year >= 1995)
    & (raw.Mileage >= 1000)
    & (raw.Mileage <= 500000)
].copy()
df["Age"] = (DATA_YEAR - df["Year"]).clip(lower=0)

# ---------- Model ----------
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X = np.hstack([encoder.fit_transform(df[CATS]), df[NUMS].values])
y = np.log(df["Price"].values)

# Older age and higher mileage can only push the price down.
MONOTONIC = [0, 0, 0, 0, 0, -1, -1, 0]


def new_model():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.07, monotonic_cst=MONOTONIC, random_state=1
    )


# Honest accuracy: predict every car using a model that never saw it.
cv_pred = cross_val_predict(new_model(), X, y, cv=KFold(5, shuffle=True, random_state=1))
cv_err = np.abs(np.exp(cv_pred) - df["Price"].values) / df["Price"].values
MEDIAN_ERROR = round(float(np.median(cv_err)) * 100)
WITHIN_25 = round(float((cv_err <= 0.25).mean()) * 100)
resid = y - cv_pred
LOW_MULT = float(np.exp(np.quantile(resid, 0.10)))
HIGH_MULT = float(np.exp(np.quantile(resid, 0.90)))

model = new_model().fit(X, y)

# ---------- Form options ----------
make_counts = df["Make"].value_counts()
MAKES = {}
for make in sorted(make_counts[make_counts >= 10].index):
    t = df[df.Make == make]["Type"].value_counts()
    MAKES[make] = sorted(t[t >= 3].index)
TYPE_ENGINE = df.groupby(["Make", "Type"])["Engine_Size"].median().round(1)
ENGINE_DEFAULTS = {m: {t: float(TYPE_ENGINE.get((m, t), 2.5)) for t in ts} for m, ts in MAKES.items()}
OPTIONS = ["Standard", "Semi Full", "Full"]
FUELS = ["Gas", "Diesel", "Hybrid"]
GEARS = ["Automatic", "Manual"]


# ---------- Maintenance cost (rule-of-thumb, NOT learned from the dataset) ----------
# The listings have no maintenance data, so this is a transparent estimate you can tune.
TIER_BASE = {"Everyday": 2000, "Mid-range": 3000, "Premium": 4000, "Luxury": 7000}   # AED per year, young car
MAKE_TIER = {m: "Everyday" for m in [
    "Toyota", "Honda", "Hyundai", "Kia", "Nissan", "Mazda", "Mitsubishi", "Suzuki", "Isuzu",
    "Changan", "Geely", "HAVAL", "MG", "Renault", "Peugeot", "Fiat"]}
MAKE_TIER.update({m: "Mid-range" for m in [
    "Chevrolet", "Ford", "GMC", "Dodge", "Chrysler", "Jeep", "Volkswagen", "MINI"]})
MAKE_TIER.update({m: "Premium" for m in ["Lexus", "Genesis", "INFINITI", "Cadillac", "Lincoln"]})
MAKE_TIER.update({m: "Luxury" for m in ["Mercedes", "BMW", "Audi", "Land Rover", "Jaguar", "Porsche"]})
MAINT_RANGE = 0.30  # +/- 30%


def aed100(v):
    return f"{CURRENCY} {round(v / 100) * 100:,.0f}"


def maintenance(f):
    tier = MAKE_TIER.get(f["Make"], "Mid-range")
    base = TIER_BASE[tier]
    age_f = 1 + 0.07 * f["Age"]                     # older cars need more
    km_f = 1 + 0.15 * (f["Mileage"] / 100000)       # high mileage wears parts
    eng_f = 1 + 0.05 * (f["Engine_Size"] - 2.0)     # bigger engines cost more to service
    year = base * min(age_f, 2.2) * km_f * max(eng_f, 0.9)
    return {
        "tier": tier,
        "year": aed100(year), "year_low": aed100(year * (1 - MAINT_RANGE)), "year_high": aed100(year * (1 + MAINT_RANGE)),
        "month": aed100(year / 12),
    }


# ---------- Helpers ----------
def aed(v):
    return f"{CURRENCY} {round(v / 500) * 500:,.0f}"


def signed(v):
    return ("+" if v >= 0 else "-") + f"{CURRENCY} {abs(round(v / 500) * 500):,.0f}"


def predict(f):
    row = pd.DataFrame([[f[c] for c in FEATURES]], columns=FEATURES)
    x = np.hstack([encoder.transform(row[CATS]), row[NUMS].values.astype(float)])
    return float(np.exp(model.predict(x)[0])) * SAR_TO_AED


def typical_mileage(age):
    for w in range(1, 6):
        sub = df[(df.Age - age).abs() <= w]
        if len(sub) >= 15:
            return float(sub["Mileage"].median())
    return 18000.0 * max(age, 1)


def explain(f):
    items = []
    final = predict(f)

    # Age and mileage: a step-by-step walk from "nearly new" to this car (sums exactly).
    new_f = dict(f, Age=0, Mileage=typical_mileage(0))
    aged_f = dict(f, Mileage=typical_mileage(f["Age"]))
    p_new, p_aged = predict(new_f), predict(aged_f)
    typ = typical_mileage(f["Age"])
    steps = [
        {"label": "Same car, nearly new", "amount": aed(p_new), "delta": None, "good": True},
        {"label": f"Age ({f['Age']} years old)", "amount": None, "delta": signed(p_aged - p_new),
         "good": p_aged >= p_new},
        {"label": f"Mileage ({f['Mileage']:,.0f} km vs ~{round(typ, -3):,.0f} km typical)",
         "amount": None, "delta": signed(final - p_aged), "good": final >= p_aged},
    ]

    # Other factors, each compared with a baseline on its own.
    other = []
    if f["Options"] != "Standard":
        d = final - predict(dict(f, Options="Standard"))
        other.append((d >= 0, f"{f['Options']} options", f"{signed(d)} compared with a Standard-spec version"))
    else:
        d = final - predict(dict(f, Options="Full"))
        other.append((d >= 0, "Standard options", f"{signed(d)} compared with a Full-spec version"))
    if f["Gear_Type"] == "Manual":
        d = final - predict(dict(f, Gear_Type="Automatic"))
        other.append((d >= 0, "Manual gearbox", f"{signed(d)} compared with an automatic"))
    eng_typ = ENGINE_DEFAULTS.get(f["Make"], {}).get(f["Type"])
    if eng_typ and abs(f["Engine_Size"] - eng_typ) >= 0.3:
        d = final - predict(dict(f, Engine_Size=eng_typ))
        other.append((d >= 0, f"{f['Engine_Size']}L engine",
                      f"{signed(d)} compared with the usual {eng_typ}L for this model"))
    if f["Fuel_Type"] != "Gas":
        d = final - predict(dict(f, Fuel_Type="Gas"))
        other.append((d >= 0, f"{f['Fuel_Type']} engine", f"{signed(d)} compared with a petrol version"))

    # Comparable listings in the data
    comp = df[(df.Make == f["Make"]) & (df.Type == f["Type"]) & ((df.Age - f["Age"]).abs() <= 2)]
    comparable = None
    if len(comp) >= 5:
        comparable = {"n": len(comp), "median": aed(float(comp["Price"].median()) * SAR_TO_AED)}

    return steps, other, comparable


@app.route("/", methods=["GET", "POST"])
def index():
    out, error, form = None, None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            make, typ = form["make"], form["type"]
            year, mileage, engine = int(form["year"]), int(form["mileage"]), float(form["engine"])
            gear, options, fuel = form["gear"], form["options"], form["fuel"]
            if make not in MAKES or typ not in MAKES[make]:
                raise ValueError
            if not (MIN_YEAR <= year <= CURRENT_YEAR) or not (0 <= mileage <= 600000) or not (1.0 <= engine <= 9.0):
                raise ValueError
            if gear not in GEARS or options not in OPTIONS or fuel not in FUELS:
                raise ValueError
            age = min(max(CURRENT_YEAR - year, 0), 26)
            f = {"Make": make, "Type": typ, "Gear_Type": gear, "Options": options, "Fuel_Type": fuel,
                 "Age": age, "Mileage": mileage, "Engine_Size": engine}
            mid = predict(f)
            steps, other, comparable = explain(f)
            out = {
                "mid": aed(mid), "low": aed(mid * LOW_MULT), "high": aed(mid * HIGH_MULT),
                "steps": steps, "other": other, "maint": maintenance(f), "comparable": comparable,
                "title": f"{year} {make} {typ}",
            }
        except Exception:
            error = "Please check the car details: pick a make and model, and enter a realistic year, mileage and engine size."
    return render_template(
        "index.html", out=out, error=error, form=form, makes=MAKES, engines=ENGINE_DEFAULTS,
        options=OPTIONS, fuels=FUELS, gears=GEARS, min_year=MIN_YEAR, max_year=CURRENT_YEAR,
        n_cars=len(df), median_error=MEDIAN_ERROR, within_25=WITHIN_25, data_year=DATA_YEAR,
        currency=CURRENCY,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
