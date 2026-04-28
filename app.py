from flask import Flask, render_template, request, redirect, session
import os
import json
from datetime import datetime
from urllib.parse import quote_plus
import uuid
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = "skincare_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
users_file = os.path.join(BASE_DIR, "users.csv")
history_file = os.path.join(BASE_DIR, "history.csv")


def _resolve_products_file():
    candidates = [
        "Produk_Skincare_with_Shopee.csv",
        "Produk_Skincare_Cleaned.csv",
        "skincare_products_300.csv",
    ]
    for filename in candidates:
        full_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError("File produk tidak ditemukan.")

if not os.path.exists(users_file):
    pd.DataFrame(columns=["username", "email", "password"]).to_csv(users_file, index=False)
if not os.path.exists(history_file):
    pd.DataFrame(
        columns=[
            "created_at",
            "email",
            "age",
            "gender",
            "hydration",
            "oil",
            "sensitivity",
            "humidity",
            "temperature",
            "skin_type",
            "concern",
            "products_json",
        ]
    ).to_csv(history_file, index=False)


def _ensure_history_schema():
    history_df = pd.read_csv(history_file)
    if "history_id" not in history_df.columns:
        history_df["history_id"] = ""
    expected_columns = [
        "history_id",
        "created_at",
        "email",
        "age",
        "gender",
        "hydration",
        "oil",
        "sensitivity",
        "humidity",
        "temperature",
        "skin_type",
        "concern",
        "products_json",
    ]
    for col in expected_columns:
        if col not in history_df.columns:
            history_df[col] = ""
    history_df = history_df[expected_columns]
    history_df.to_csv(history_file, index=False)


_ensure_history_schema()

skin_df_raw = pd.read_csv(os.path.join(BASE_DIR, "Skin_Type_OG.csv"))
products = pd.read_csv(_resolve_products_file())
interactions = pd.read_csv(os.path.join(BASE_DIR, "interactions_300.csv"))
users_300 = pd.read_csv(os.path.join(BASE_DIR, "users_300.csv"))

skin_df_raw["Skin_Type"] = skin_df_raw["Skin_Type"].astype(str).str.strip()
products["skin_type"] = products["skin_type"].astype(str).str.strip()
products["category"] = products["category"].astype(str).str.strip()
products["concern"] = products["concern"].astype(str).str.strip()
if "product_name" not in products.columns and "name" in products.columns:
    products["product_name"] = products["name"]
users_300["skin_type"] = users_300["skin_type"].astype(str).str.strip()
users_300["concern"] = users_300["concern"].astype(str).str.strip()

link_column = None
for candidate in ["buy_link", "product_link", "url", "link", "shopee_link"]:
    if candidate in products.columns:
        link_column = candidate
        break

if link_column:
    products["buy_link"] = products[link_column].astype(str).str.strip()
    products.loc[
        products["buy_link"].str.lower().isin(["", "nan", "none"]),
        "buy_link",
    ] = ""
else:
    products["buy_link"] = ""

products["buy_link"] = products["buy_link"].astype(str).str.strip()
products.loc[products["buy_link"].str.lower().isin(["nan", "none"]), "buy_link"] = ""
products.loc[
    (products["buy_link"] != "") & (~products["buy_link"].str.match(r"^https?://", na=False)),
    "buy_link",
] = "https://" + products.loc[
    (products["buy_link"] != "") & (~products["buy_link"].str.match(r"^https?://", na=False)),
    "buy_link",
]

missing_links = products["buy_link"].astype(str).str.len() == 0
products.loc[missing_links, "buy_link"] = products.loc[missing_links, "product_name"].apply(
    lambda name: f"https://shopee.co.id/search?keyword={quote_plus(str(name))}"
)

skin_df = skin_df_raw.copy()

le_skin = LabelEncoder()
skin_df["Skin_Type"] = le_skin.fit_transform(skin_df["Skin_Type"])
le_gender = LabelEncoder()
skin_df["Gender"] = le_gender.fit_transform(skin_df["Gender"])
le_hydration = LabelEncoder()
skin_df["Hydration_Level"] = le_hydration.fit_transform(skin_df["Hydration_Level"])
le_oil = LabelEncoder()
skin_df["Oil_Level"] = le_oil.fit_transform(skin_df["Oil_Level"])
le_sens = LabelEncoder()
skin_df["Sensitivity"] = le_sens.fit_transform(skin_df["Sensitivity"])

X = skin_df[["Age", "Gender", "Hydration_Level", "Oil_Level", "Sensitivity", "Humidity", "Temperature"]]
y = skin_df["Skin_Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_skin = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model_skin.fit(X_train, y_train)

interaction_with_user_profile = interactions.merge(users_300, on="user_id", how="left")

# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    hero_images = [
        "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1556228720-195a672e8a03?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1556228453-efd6c1ff04f6?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?auto=format&fit=crop&w=600&q=80",
    ]
    return render_template("index.html", hero_images=hero_images)


@app.route("/get-started")
def get_started():
    if "user" in session:
        return redirect("/form")
    return redirect("/register")


# =====================================================
# REGISTER
# =====================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        users_df = pd.read_csv(users_file)
        users_df["email"] = users_df["email"].astype(str).str.strip().str.lower()
        users_df["password"] = users_df["password"].astype(str).str.strip()
        if email in users_df["email"].values:
            return render_template("register.html", error="Email sudah terdaftar.")

        new_user = pd.DataFrame([{
            "username": username,
            "email": email,
            "password": password
        }])

        users_df = pd.concat(
            [users_df, new_user],
            ignore_index=True
        )

        users_df.to_csv(users_file, index=False)

        return redirect("/login")
    return render_template("register.html")


# =====================================================
# LOGIN
# =====================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        users_df = pd.read_csv(users_file)

        users_df["email"] = users_df["email"].astype(str).str.strip().str.lower()
        users_df["password"] = users_df["password"].astype(str).str.strip()

        user = users_df[(users_df["email"] == email) & (users_df["password"] == password)]

        if not user.empty:
            session["user"] = email
            return redirect("/form")
        return render_template("login.html", error="Login gagal, email atau password salah.")
    return render_template("login.html")


# =====================================================
# LOGOUT
# =====================================================

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# =====================================================
# FORM (PROTECTED PAGE)
# =====================================================

@app.route("/form")
def form():
    if "user" not in session:
        return redirect("/register")
    return render_template("form.html")


# =====================================================
# PROFILE ACCOUNT
# =====================================================

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect("/login")

    users_df = pd.read_csv(users_file)

    users_df["email"] = (
        users_df["email"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    current_email = (
        session["user"]
        .strip()
        .lower()
    )

    user_data = users_df[
        users_df["email"] == current_email
    ]

    if user_data.empty:
        return redirect("/logout")

    username = user_data.iloc[0]["username"]
    email = user_data.iloc[0]["email"]

    return render_template(
        "profile.html",
        username=username,
        email=email
    )


@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    current_email = session["user"].strip().lower()
    history_df = pd.read_csv(history_file)

    if history_df.empty:
        return render_template("history.html", histories=[])

    history_df["email"] = history_df["email"].astype(str).str.strip().str.lower()
    user_history = history_df[history_df["email"] == current_email].copy()
    user_history = user_history.sort_values(by="created_at", ascending=False)

    histories = []
    for _, row in user_history.iterrows():
        try:
            product_rows = json.loads(row["products_json"])
        except Exception:
            product_rows = []

        histories.append({
            "history_id": row.get("history_id", ""),
            "created_at": row.get("created_at", "-"),
            "age": row.get("age", "-"),
            "gender": row.get("gender", "-"),
            "hydration": row.get("hydration", "-"),
            "oil": row.get("oil", "-"),
            "sensitivity": row.get("sensitivity", "-"),
            "humidity": row.get("humidity", "-"),
            "temperature": row.get("temperature", "-"),
            "skin_type": row.get("skin_type", "-"),
            "concern": row.get("concern", "-"),
            "products": product_rows,
        })

    return render_template("history.html", histories=histories)


@app.route("/history/delete/<history_id>", methods=["POST"])
def delete_history(history_id):
    if "user" not in session:
        return redirect("/login")

    history_df = pd.read_csv(history_file)
    current_email = session["user"].strip().lower()
    history_df["email"] = history_df["email"].astype(str).str.strip().str.lower()
    history_df["history_id"] = history_df["history_id"].astype(str)

    filtered = history_df[
        ~(
            (history_df["email"] == current_email)
            & (history_df["history_id"] == str(history_id))
        )
    ]
    filtered.to_csv(history_file, index=False)
    return redirect("/history")


@app.route("/history/clear", methods=["POST"])
def clear_history():
    if "user" not in session:
        return redirect("/login")

    history_df = pd.read_csv(history_file)
    current_email = session["user"].strip().lower()
    history_df["email"] = history_df["email"].astype(str).str.strip().str.lower()
    history_df = history_df[history_df["email"] != current_email]
    history_df.to_csv(history_file, index=False)
    return redirect("/history")


def _safe_encode(encoder, value):
    classes = set(encoder.classes_)
    return encoder.transform([value if value in classes else encoder.classes_[0]])[0]


def _dominant_concern(age, skin_type):
    pool = users_300[
        (users_300["skin_type"].str.lower() == skin_type.lower())
        & (users_300["age"].between(age - 5, age + 5))
    ]
    if pool.empty:
        pool = users_300[users_300["skin_type"].str.lower() == skin_type.lower()]
    if pool.empty:
        return "General Care"
    return pool["concern"].mode().iloc[0]


def _score_products(age, skin_type, concern):
    profile_pool = users_300.copy()
    profile_pool["age_diff"] = (profile_pool["age"] - age).abs()
    profile_pool["skin_match"] = profile_pool["skin_type"].str.lower() == skin_type.lower()

    candidate_users = profile_pool.sort_values(
        by=["skin_match", "age_diff"], ascending=[False, True]
    ).head(40)["user_id"].tolist()

    cf = (
        interaction_with_user_profile[
            interaction_with_user_profile["user_id"].isin(candidate_users)
        ]
        .groupby("product_id")["rating"]
        .mean()
        .rename("cf_score")
    )

    scored = products.copy()
    scored["skin_score"] = (
        scored["skin_type"].str.lower() == skin_type.lower()
    ).astype(float)
    scored["concern_score"] = (
        scored["concern"].str.lower() == concern.lower()
    ).astype(float)
    scored["cf_score"] = scored["product_id"].map(cf).fillna(0.0)
    max_cf = scored["cf_score"].max()
    if max_cf > 0:
        scored["cf_score"] = scored["cf_score"] / max_cf

    scored["final_score"] = (
        0.45 * scored["skin_score"] + 0.35 * scored["concern_score"] + 0.20 * scored["cf_score"]
    )
    return scored.sort_values(by="final_score", ascending=False)


def _pick_one_per_routine(scored_products):
    routine_slots = {
        "Cleanser": {"facial wash", "facial cleanser", "sabun", "cleanser", "soap"},
        "Prep/Treatment": {"toner", "essence", "exfoliator", "acne treatment"},
        "Serum": {"serum", "ampoule"},
        "Moisturizer": {"moisturizer", "cream", "gel", "lotion", "emulsion"},
        "Sunscreen": {"sunscreen", "sunblock", "spf"},
    }

    selected = []
    selected_ids = set()

    for slot_name, aliases in routine_slots.items():
        slot_candidates = scored_products[
            scored_products["category"].str.lower().apply(
                lambda c: any(keyword in c for keyword in aliases)
            )
        ]
        if not slot_candidates.empty:
            chosen = slot_candidates.iloc[0].copy()
            chosen["routine_step"] = slot_name
            selected.append(chosen)
            selected_ids.add(chosen["product_id"])

    if len(selected) < 5:
        for _, row in scored_products.iterrows():
            if row["product_id"] in selected_ids:
                continue
            extra = row.copy()
            extra["routine_step"] = "Tambahan"
            selected.append(extra)
            selected_ids.add(row["product_id"])
            if len(selected) == 5:
                break

    return pd.DataFrame(selected).head(5)


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/register")

    age = int(request.form["age"])
    gender = request.form["gender"]
    hydration = request.form["hydration"]
    oil = request.form["oil"]
    sensitivity = request.form["sensitivity"]
    humidity = float(request.form["humidity"])
    temperature = float(request.form["temperature"])

    gender_encoded = _safe_encode(le_gender, gender)
    hydration_encoded = _safe_encode(le_hydration, hydration)
    oil_encoded = _safe_encode(le_oil, oil)
    sensitivity_encoded = _safe_encode(le_sens, sensitivity)

    input_data = [[
        age,
        gender_encoded,
        hydration_encoded,
        oil_encoded,
        sensitivity_encoded,
        humidity,
        temperature
    ]]

    pred = model_skin.predict(input_data)[0]
    skin_type = le_skin.inverse_transform([pred])[0]
    concern = _dominant_concern(age, skin_type)

    scored_products = _score_products(age, skin_type, concern)
    result = _pick_one_per_routine(scored_products)
    result_records = result.to_dict(orient="records")

    history_df = pd.read_csv(history_file)
    new_history = pd.DataFrame([{
        "history_id": str(uuid.uuid4()),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email": session["user"].strip().lower(),
        "age": age,
        "gender": gender,
        "hydration": hydration,
        "oil": oil,
        "sensitivity": sensitivity,
        "humidity": humidity,
        "temperature": temperature,
        "skin_type": skin_type,
        "concern": concern,
        "products_json": json.dumps(result_records),
    }])
    history_df = pd.concat([history_df, new_history], ignore_index=True)
    history_df.to_csv(history_file, index=False)

    carousel_products = products.head(20).to_dict(orient="records")

    return render_template(
        "result.html",
        skin_type=skin_type,
        concern=concern,
        products=result_records,
        carousel_products=carousel_products
    )

if __name__ == "__main__":
    app.run(debug=True)