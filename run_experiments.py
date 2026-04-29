import json
import math
import os
import random
from itertools import product

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "report_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)


def resolve_products_file():
    for name in ["Produk_Skincare_with_Shopee.csv", "Produk_Skincare_Cleaned.csv", "skincare_products_300.csv"]:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Product CSV not found")


def load_data():
    products = pd.read_csv(resolve_products_file())
    interactions = pd.read_csv(os.path.join(BASE_DIR, "interactions_300.csv"))
    if "product_name" not in products.columns and "name" in products.columns:
        products["product_name"] = products["name"]
    for col in ["category", "concern", "skin_type", "product_name"]:
        products[col] = products[col].astype(str).str.strip()
    return products, interactions


def split_data(interactions):
    train_parts, test_parts = [], []
    for _, g in interactions.groupby("user_id"):
        if len(g) < 2:
            continue
        idx = list(g.index)
        random.shuffle(idx)
        test_idx = idx[0]
        test_parts.append(g.loc[[test_idx]])
        train_parts.append(g.drop(test_idx))
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def build_features(products, w):
    c = pd.get_dummies(products["category"]) * w["w_category"]
    con = pd.get_dummies(products["concern"]) * w["w_concern"]
    s = pd.get_dummies(products["skin_type"]) * w["w_skin_type"]
    p = pd.to_numeric(products["price_numeric"], errors="coerce").fillna(0.0)
    p = (p - p.min()) / (p.max() - p.min()) if p.max() > p.min() else p * 0
    p = pd.DataFrame({"price": p * w["w_price"]})
    x = pd.concat([c, con, s, p], axis=1).to_numpy(dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    x = x / n
    return x, {int(pid): i for i, pid in enumerate(products["product_id"].tolist())}


def cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def pred_cbf(user, item, train, feats, idx_map):
    hist = train[train["user_id"] == user]
    if hist.empty or item not in idx_map:
        return 3.0
    tv = feats[idx_map[item]]
    sims, rates = [], []
    for _, r in hist.iterrows():
        pid = int(r["product_id"])
        if pid in idx_map:
            sims.append(max(cos(tv, feats[idx_map[pid]]), 0))
            rates.append(float(r["rating"]))
    if not sims:
        return 3.0
    sims = np.array(sims)
    rates = np.array(rates)
    return float(np.dot(sims, rates) / sims.sum()) if sims.sum() > 0 else float(rates.mean())


def build_cf(train):
    m = train.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0.0)
    a = m.to_numpy(dtype=float)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1
    an = a / n
    sim = np.dot(an, an.T)
    np.fill_diagonal(sim, 0.0)
    return m, sim


def pred_cf(user, item, m, sim, p):
    if user not in m.index or item not in m.columns:
        return 3.0
    upos = m.index.get_loc(user)
    sims = sim[upos]
    col = m[item].to_numpy(dtype=float)
    mask = col > 0
    if not np.any(mask):
        return 3.0
    sims, rates = sims[mask], col[mask]
    k = min(p["k_neighbors"], len(sims))
    top = np.argsort(sims)[-k:]
    st, rt = np.maximum(sims[top], 0), rates[top]
    den = st.sum() + p["shrinkage"]
    return float(np.dot(st, rt) / den) if den > 0 else float(rt.mean())


def eval_model(name, train, test, products, cbf_p, cf_p, alpha):
    feats, idx = build_features(products, cbf_p)
    m, sim = build_cf(train)
    yt, yp = [], []
    all_items = set(products["product_id"].astype(int).tolist())
    user_train = train.groupby("user_id")["product_id"].apply(lambda s: set(s.astype(int))).to_dict()
    user_test = test.groupby("user_id")["product_id"].apply(lambda s: set(s.astype(int))).to_dict()
    p5, r5, n5 = [], [], []

    for _, row in test.iterrows():
        u, i, y = int(row["user_id"]), int(row["product_id"]), float(row["rating"])
        c = pred_cbf(u, i, train, feats, idx)
        f = pred_cf(u, i, m, sim, cf_p)
        pred = c if name == "CBF" else (f if name == "CF" else alpha * c + (1 - alpha) * f)
        yt.append(y)
        yp.append(pred)

    eval_users = sorted(user_test.keys())[:40]
    for u in eval_users:
        gt = user_test[u]
        seen = user_train.get(u, set())
        cand = list(all_items - seen)
        if len(cand) > 120:
            random.shuffle(cand)
            cand = cand[:120]
        scored = []
        for i in cand:
            c = pred_cbf(u, i, train, feats, idx)
            f = pred_cf(u, i, m, sim, cf_p)
            s = c if name == "CBF" else (f if name == "CF" else alpha * c + (1 - alpha) * f)
            scored.append((i, s))
        top = [i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:5]]
        hit = [1 if i in gt else 0 for i in top]
        p5.append(sum(hit) / 5.0)
        r5.append(sum(hit) / max(len(gt), 1))
        dcg = sum(((2**h - 1) / math.log2(k + 2)) for k, h in enumerate(hit))
        idcg = 1.0
        n5.append(dcg / idcg)

    yt = np.array(yt)
    yp = np.array(yp)
    return {
        "model": name,
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE": float(np.mean(np.abs(yt - yp))),
        "Precision@5": float(np.mean(p5)),
        "Recall@5": float(np.mean(r5)),
        "NDCG@5": float(np.mean(n5)),
    }


def train_mf(train, n_factors=20, lr=0.01, reg=0.05, epochs=15, batch_size=256):
    users = sorted(train["user_id"].astype(int).unique().tolist())
    items = sorted(train["product_id"].astype(int).unique().tolist())
    u2i = {u: i for i, u in enumerate(users)}
    p2i = {p: i for i, p in enumerate(items)}
    n_users, n_items = len(users), len(items)
    P = 0.1 * np.random.randn(n_users, n_factors)
    Q = 0.1 * np.random.randn(n_items, n_factors)
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    mu = float(train["rating"].mean())

    data = train[["user_id", "product_id", "rating"]].copy()
    rows = data.to_records(index=False)
    for _ in range(epochs):
        idx = np.arange(len(rows))
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch = idx[start : start + batch_size]
            for b in batch:
                u, it, r = rows[b]
                u = u2i[int(u)]
                it = p2i[int(it)]
                pred = mu + bu[u] + bi[it] + np.dot(P[u], Q[it])
                err = float(r) - pred
                bu[u] += lr * (err - reg * bu[u])
                bi[it] += lr * (err - reg * bi[it])
                pu = P[u].copy()
                qi = Q[it].copy()
                P[u] += lr * (err * qi - reg * pu)
                Q[it] += lr * (err * pu - reg * qi)

    return {"P": P, "Q": Q, "bu": bu, "bi": bi, "mu": mu, "u2i": u2i, "p2i": p2i}


def pred_mf(model, user, item):
    if user not in model["u2i"] or item not in model["p2i"]:
        return model["mu"]
    u = model["u2i"][user]
    it = model["p2i"][item]
    return float(model["mu"] + model["bu"][u] + model["bi"][it] + np.dot(model["P"][u], model["Q"][it]))


def eval_mf(train, test, products, mf_params):
    model = train_mf(
        train,
        n_factors=mf_params["n_factors"],
        lr=mf_params["learning_rate"],
        reg=mf_params["reg"],
        epochs=mf_params["epochs"],
        batch_size=mf_params["batch_size"],
    )
    yt = test["rating"].astype(float).to_numpy()
    yp = np.array([pred_mf(model, int(u), int(i)) for u, i in zip(test["user_id"], test["product_id"])])
    all_items = set(products["product_id"].astype(int).tolist())
    user_train = train.groupby("user_id")["product_id"].apply(lambda s: set(s.astype(int))).to_dict()
    user_test = test.groupby("user_id")["product_id"].apply(lambda s: set(s.astype(int))).to_dict()

    p5, r5, n5 = [], [], []
    eval_users = sorted(user_test.keys())[:40]
    for u in eval_users:
        gt = user_test[u]
        seen = user_train.get(u, set())
        cand = list(all_items - seen)
        if len(cand) > 120:
            random.shuffle(cand)
            cand = cand[:120]
        scored = [(i, pred_mf(model, int(u), int(i))) for i in cand]
        top = [i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:5]]
        hit = [1 if i in gt else 0 for i in top]
        p5.append(sum(hit) / 5.0)
        r5.append(sum(hit) / max(len(gt), 1))
        dcg = sum(((2**h - 1) / math.log2(k + 2)) for k, h in enumerate(hit))
        n5.append(dcg / 1.0)

    return {
        "model": "MF-SGD",
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE": float(np.mean(np.abs(yt - yp))),
        "Precision@5": float(np.mean(p5)),
        "Recall@5": float(np.mean(r5)),
        "NDCG@5": float(np.mean(n5)),
    }


def main():
    products, inter = load_data()
    train, test = split_data(inter)

    cbf_grid = [
        {"w_category": 1.0, "w_concern": 1.0, "w_skin_type": 1.0, "w_price": 0.05},
        {"w_category": 1.2, "w_concern": 1.0, "w_skin_type": 1.0, "w_price": 0.05},
        {"w_category": 1.0, "w_concern": 1.3, "w_skin_type": 1.0, "w_price": 0.05},
    ]
    cf_candidates = [{"k_neighbors": 15, "shrinkage": 0.05}, {"k_neighbors": 30, "shrinkage": 0.2}]
    alphas = [0.4, 0.6]

    best = {"score": -1}
    tuning_rows = []
    for c in cbf_grid:
        for f in cf_candidates:
            for a in alphas:
                h = eval_model("Hybrid", train, test, products, c, f, a)
                score = h["NDCG@5"] + h["Precision@5"]
                row = {**c, **f, "alpha": a, **h}
                tuning_rows.append(row)
                if score > best["score"]:
                    best = {"score": score, "cbf": c, "cf": f, "alpha": a}

    cbf_res = eval_model("CBF", train, test, products, best["cbf"], best["cf"], best["alpha"])
    cf_res = eval_model("CF", train, test, products, best["cbf"], best["cf"], best["alpha"])
    hy_res = eval_model("Hybrid", train, test, products, best["cbf"], best["cf"], best["alpha"])
    mf_grid = [
        {"n_factors": 12, "learning_rate": 0.01, "reg": 0.05, "epochs": 12, "batch_size": 256},
        {"n_factors": 20, "learning_rate": 0.008, "reg": 0.05, "epochs": 15, "batch_size": 256},
        {"n_factors": 24, "learning_rate": 0.006, "reg": 0.08, "epochs": 18, "batch_size": 512},
    ]
    mf_rows = []
    best_mf = None
    for p in mf_grid:
        res = eval_mf(train, test, products, p)
        row = {**p, **res}
        mf_rows.append(row)
        if best_mf is None or (res["NDCG@5"] + res["Precision@5"]) > (best_mf["NDCG@5"] + best_mf["Precision@5"]):
            best_mf = row

    tuning_df = pd.DataFrame(tuning_rows).sort_values(by=["NDCG@5", "Precision@5"], ascending=False)
    metrics_df = pd.DataFrame([cbf_res, cf_res, hy_res, {
        "model": best_mf["model"],
        "RMSE": best_mf["RMSE"],
        "MAE": best_mf["MAE"],
        "Precision@5": best_mf["Precision@5"],
        "Recall@5": best_mf["Recall@5"],
        "NDCG@5": best_mf["NDCG@5"],
    }]).sort_values(by=["NDCG@5", "Precision@5"], ascending=False)
    mf_tuning_df = pd.DataFrame(mf_rows).sort_values(by=["NDCG@5", "Precision@5"], ascending=False)

    tuning_df.to_csv(os.path.join(OUT_DIR, "tuning_results.csv"), index=False)
    mf_tuning_df.to_csv(os.path.join(OUT_DIR, "mf_tuning_results.csv"), index=False)
    metrics_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)
    with open(os.path.join(OUT_DIR, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with open(os.path.join(OUT_DIR, "best_mf_params.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_factors": best_mf["n_factors"],
                "learning_rate": best_mf["learning_rate"],
                "reg": best_mf["reg"],
                "epochs": best_mf["epochs"],
                "batch_size": best_mf["batch_size"],
            },
            f,
            indent=2,
        )

    print("Done. Artifacts in report_artifacts/")


if __name__ == "__main__":
    main()
