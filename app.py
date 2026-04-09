from flask import Flask, render_template, request
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ======================
# LOAD DATA
# ======================
skin_df = pd.read_csv("Skin_Type_OG.csv")
products = pd.read_csv("skincare_products_300.csv")
interactions = pd.read_csv("interactions_300.csv")

print(skin_df["Skin_Type"].value_counts())

# ======================
# PREPROCESSING
# ======================

# encode target
le_skin = LabelEncoder()
skin_df["Skin_Type"] = le_skin.fit_transform(skin_df["Skin_Type"])

# encode fitur
le_gender = LabelEncoder()
skin_df["Gender"] = le_gender.fit_transform(skin_df["Gender"])

le_hydration = LabelEncoder()
skin_df["Hydration_Level"] = le_hydration.fit_transform(skin_df["Hydration_Level"])

le_oil = LabelEncoder()
skin_df["Oil_Level"] = le_oil.fit_transform(skin_df["Oil_Level"])

le_sens = LabelEncoder()
skin_df["Sensitivity"] = le_sens.fit_transform(skin_df["Sensitivity"])

# fitur & target
X = skin_df[[
    "Age","Gender","Hydration_Level","Oil_Level",
    "Sensitivity","Humidity","Temperature"
]]
y = skin_df["Skin_Type"]

# training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_skin = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
model_skin.fit(X_train, y_train)

# ======================
# COLLABORATIVE FILTERING
# ======================
user_item = interactions.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating'
).fillna(0)

# ======================
# ROUTES
# ======================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/form')
def form():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():

    # ======================
    # INPUT
    # ======================
    age = int(request.form['age'])
    gender = request.form['gender']
    hydration = request.form['hydration']
    oil = request.form['oil']
    sensitivity = request.form['sensitivity']
    humidity = float(request.form['humidity'])
    temperature = float(request.form['temperature'])

    # ======================
    # ENCODE
    # ======================
    gender = le_gender.transform([gender])[0]
    hydration = le_hydration.transform([hydration])[0]
    oil = le_oil.transform([oil])[0]
    sensitivity = le_sens.transform([sensitivity])[0]

    # ======================
    # PREDIKSI ML
    # ======================
    input_data = [[age, gender, hydration, oil, sensitivity, humidity, temperature]]
    pred = model_skin.predict(input_data)[0]
    skin_type = le_skin.inverse_transform([pred])[0]

    # ======================
    # RULE-BASED FIX (WAJIB)
    # ======================
    if hydration == 0 and oil == 0:
        skin_type = "Dry"

    elif hydration == 2 and oil == 0:
        skin_type = "Dry"

    elif oil == 2:
        skin_type = "Oily"

    elif hydration == 1 and oil == 1:
        skin_type = "Combination"

    # ======================
    # FILTER PRODUK
    # ======================
    filtered_products = products[
        products['skin_type'] == skin_type
    ].copy()

    if filtered_products.empty:
        filtered_products = products.copy()

    # ======================
    # CONTENT SCORE
    # ======================
    filtered_products['content_score'] = 1

    # ======================
    # COLLABORATIVE
    # ======================
    user_id = random.randint(1, 300)

    if user_id in user_item.index:
        user_vector = user_item.loc[user_id].values.reshape(1, -1)
        sim_scores = cosine_similarity(user_vector, user_item)[0]

        similar_users = sim_scores.argsort()[-5:]
        cf_scores = user_item.iloc[similar_users].mean(axis=0)

        filtered_products['cf_score'] = filtered_products['product_id'].map(cf_scores).fillna(0)
    else:
        filtered_products['cf_score'] = 0

    # ======================
    # HYBRID
    # ======================
    filtered_products['final_score'] = (
        0.7 * filtered_products['content_score'] +
        0.3 * filtered_products['cf_score']
    )

    result = filtered_products.sort_values(
        'final_score', ascending=False
    ).head(5)

    # ======================
    # OUTPUT
    # ======================
    return render_template(
        "result.html",
        skin_type=skin_type,
        products=result.to_dict(orient="records")
    )

# ======================
# RUN
# ======================
if __name__ == '__main__':
    app.run(debug=True)