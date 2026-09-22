"""
dashboard.py  ──  House Price Prediction Dashboard
Run with:  streamlit run dashboard.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import streamlit as st

from src.generate_dataset import generate_housing_data
from src.features import add_features, get_feature_lists
from src.pipeline import build_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700; color: #1f3c88;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem; color: #555; text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f3c88 0%, #4a90d9 100%);
        border-radius: 12px; padding: 18px 24px; color: white; margin: 6px 0;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .metric-card h1 { margin: 4px 0 0; font-size: 1.9rem; }
    .predict-box {
        background: #f0f7ff; border: 2px solid #4a90d9;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .predict-price {
        font-size: 2.8rem; font-weight: 800; color: #1f3c88;
    }
    .tag { background:#e8f0fe; color:#1a73e8; border-radius:20px;
           padding:3px 10px; font-size:0.78rem; margin:2px; display:inline-block; }
    div[data-testid="stSidebar"] { background: #f8f9ff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Data + Model caching
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    path = "data/houses.csv"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        df = generate_housing_data()
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    return add_features(df)


@st.cache_resource(show_spinner="Training models… (first run only)")
def train_models(df):
    NUM, CAT = get_feature_lists()
    X = df[NUM + CAT]
    y = df["SalePrice"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    specs = {
        "Linear Regression":   Pipeline([("pre", build_preprocessor()), ("m", LinearRegression())]),
        "Ridge Regression":    Pipeline([("pre", build_preprocessor()), ("m", Ridge(alpha=10.0))]),
        "Decision Tree":       Pipeline([("pre", build_preprocessor()), ("m", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42))]),
        "Random Forest":       Pipeline([("pre", build_preprocessor()), ("m", RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1))]),
        "Gradient Boosting":   Pipeline([("pre", build_preprocessor()), ("m", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42))]),
    }

    trained, metrics = {}, {}
    for name, pipe in specs.items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        trained[name] = pipe
        metrics[name] = {
            "MAE":  mean_absolute_error(yte, pred),
            "RMSE": np.sqrt(mean_squared_error(yte, pred)),
            "R2":   r2_score(yte, pred),
        }

    # Best by RMSE
    best_name = min(metrics, key=lambda k: metrics[k]["RMSE"])
    return trained, metrics, best_name, (Xte, yte)


# ─────────────────────────────────────────────
#  Load
# ─────────────────────────────────────────────
df = load_data()
trained_models, metrics, best_name, (Xte, yte) = train_models(df)

NEIGHBORHOODS = sorted(df["Neighborhood"].unique().tolist())
ZONES         = sorted(df["MSZoning"].unique().tolist())
BLDG_TYPES    = sorted(df["BldgType"].unique().tolist())
HOUSE_STYLES  = sorted(df["HouseStyle"].unique().tolist())
KITCHEN_QUALS = ["Ex", "Gd", "TA", "Fa"]
FOUNDATIONS   = sorted(df["Foundation"].unique().tolist())

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/house--v1.png", width=64)
    st.markdown("## 🏠 House Price Predictor")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Predict Price",
        "📊 Data Explorer",
        "📈 Model Performance",
        "🔍 Feature Insights",
        "📋 Dataset Preview",
        "ℹ️ About Project"
    ])
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} houses")
    st.markdown(f"**Best Model:** {best_name}")
    st.markdown(f"**Best R²:** `{metrics[best_name]['R2']:.4f}`")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: Predict Price
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Predict Price":
    st.markdown('<div class="main-header">🏠 House Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter property details to get an instant ML-powered price estimate</div>', unsafe_allow_html=True)

    # Model selector
    col_sel, _ = st.columns([2, 3])
    with col_sel:
        chosen_model = st.selectbox("🤖 Choose Prediction Model", list(trained_models.keys()),
                                     index=list(trained_models.keys()).index(best_name))

    st.markdown("---")

    # ── Input form ──────────────────────────────────────────────────────────
    st.subheader("🏡 Property Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📐 Size & Layout**")
        lot_area    = st.number_input("Lot Area (sq ft)", 1000, 25000, 6000, 500)
        gr_liv_area = st.number_input("Above-Ground Living Area (sq ft)", 400, 5000, 1600, 100)
        total_bsmt  = st.number_input("Basement Area (sq ft)", 0, 3000, 800, 100)
        first_flr   = st.number_input("1st Floor SF", 300, 4000, 900, 100)
        second_flr  = st.number_input("2nd Floor SF", 0, 3000, 700, 100)
        tot_rooms   = st.slider("Total Rooms Above Ground", 3, 14, 7)

    with c2:
        st.markdown("**🛏️ Rooms & Quality**")
        bedrooms    = st.slider("Bedrooms", 1, 8, 3)
        full_bath   = st.slider("Full Bathrooms", 1, 4, 2)
        half_bath   = st.slider("Half Bathrooms", 0, 2, 1)
        fireplaces  = st.slider("Fireplaces", 0, 4, 1)
        overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 7)
        overall_cond = st.slider("Overall Condition (1–9)", 1, 9, 5)

    with c3:
        st.markdown("**📍 Location & Extras**")
        neighborhood  = st.selectbox("Neighborhood", NEIGHBORHOODS, index=0)
        ms_zoning     = st.selectbox("Zoning", ZONES, index=0)
        bldg_type     = st.selectbox("Building Type", BLDG_TYPES, index=0)
        house_style   = st.selectbox("House Style", HOUSE_STYLES, index=0)
        kitchen_qual  = st.selectbox("Kitchen Quality", KITCHEN_QUALS, index=1)
        foundation    = st.selectbox("Foundation", FOUNDATIONS, index=0)
        year_built    = st.slider("Year Built", 1960, 2024, 2005)
        year_remod    = st.slider("Year Remodelled", year_built, 2024, max(year_built, 2010))
        garage_cars   = st.slider("Garage Cars", 0, 4, 2)
        garage_area   = st.number_input("Garage Area (sq ft)", 0, 1200, garage_cars * 200, 50)
        paved_drive   = st.selectbox("Paved Drive", ["Y", "N"], index=0)
        central_air   = st.selectbox("Central Air", ["Y", "N"], index=0)

    st.markdown("---")

    # ── Predict ─────────────────────────────────────────────────────────────
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        NUM, CAT = get_feature_lists()
        input_row = {
            "MSZoning": ms_zoning, "LotArea": lot_area, "Neighborhood": neighborhood,
            "BldgType": bldg_type, "HouseStyle": house_style, "OverallQual": overall_qual,
            "OverallCond": overall_cond, "YearBuilt": year_built, "YearRemodAdd": year_remod,
            "Foundation": foundation, "TotalBsmtSF": float(total_bsmt),
            "1stFlrSF": first_flr, "2ndFlrSF": second_flr, "GrLivArea": gr_liv_area,
            "FullBath": full_bath, "HalfBath": half_bath, "BedroomAbvGr": bedrooms,
            "TotRmsAbvGrd": tot_rooms, "Fireplaces": fireplaces,
            "GarageCars": float(garage_cars), "GarageArea": float(garage_area),
            "PavedDrive": paved_drive, "CentralAir": central_air,
            "KitchenQual": kitchen_qual,
        }
        inp = pd.DataFrame([input_row])
        inp = add_features(inp)
        pipe = trained_models[chosen_model]
        pred_price = pipe.predict(inp[NUM + CAT])[0]

        # Confidence range (±8%)
        low  = pred_price * 0.92
        high = pred_price * 1.08

        st.markdown("---")
        st.markdown(f"""
        <div class="predict-box">
            <p style="font-size:1.1rem; color:#555; margin-bottom:6px;">🏷️ Estimated Market Value</p>
            <div class="predict-price">₹ {pred_price:,.0f}</div>
            <p style="color:#888; margin-top:6px;">Confidence Range: ₹{low:,.0f} — ₹{high:,.0f}</p>
            <p style="color:#1a73e8; font-size:0.85rem;">Model: {chosen_model}</p>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        st.markdown("#### 📌 Property Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price per sq ft", f"₹{pred_price/gr_liv_area:,.0f}")
        m2.metric("Living Area",     f"{gr_liv_area:,} sq ft")
        m3.metric("Bedrooms",        bedrooms)
        m4.metric("Overall Quality", f"{overall_qual}/10")

        # What-if analysis
        st.markdown("#### 🔧 What-If Analysis")
        st.markdown("_How price changes with quality upgrades_")
        what_if_data = []
        for q in range(max(1, overall_qual-2), min(11, overall_qual+3)):
            row2 = inp.copy()
            row2["OverallQual"] = q
            row2 = add_features(row2.drop(columns=[c for c in ["Age","RemodAge","WasRemodeled",
                "BathsTotal","RoomsPerArea","GarageQualCars","TotalSF"] if c in row2.columns]))
            row2 = add_features(row2)
            p2 = pipe.predict(row2[NUM + CAT])[0]
            what_if_data.append({"Quality": q, "Predicted Price": p2})

        wdf = pd.DataFrame(what_if_data)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(wdf["Quality"], wdf["Predicted Price"] / 1e6, marker="o", color="#1f3c88", lw=2)
        ax.axvline(overall_qual, color="red", linestyle="--", alpha=0.6, label=f"Current (Q={overall_qual})")
        ax.set_xlabel("Overall Quality")
        ax.set_ylabel("Predicted Price (₹M)")
        ax.set_title("Price Sensitivity to Quality")
        ax.legend()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<div class="main-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore distributions and relationships in the housing dataset</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Neighborhood Analysis", "Feature Correlations", "Scatter Plots"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df["SalePrice"] / 1e6, bins=40, color="#4C72B0", edgecolor="white")
            ax.set_xlabel("Sale Price (₹ Millions)")
            ax.set_title("Price Distribution")
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(np.log1p(df["SalePrice"]), bins=40, color="#DD8452", edgecolor="white")
            ax.set_xlabel("log(1 + Price)")
            ax.set_title("Log-Price Distribution")
            st.pyplot(fig); plt.close()

        st.markdown("#### Summary Statistics")
        stats = df["SalePrice"].describe().to_frame().T
        stats.columns = ["Count","Mean","Std","Min","25%","50%","75%","Max"]
        for col in ["Mean","Std","Min","25%","50%","75%","Max"]:
            stats[col] = stats[col].apply(lambda v: f"₹{v:,.0f}")
        st.dataframe(stats, use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(11, 5))
        med = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False)
        colors = sns.color_palette("muted", len(med))
        med.div(1e6).plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Median Sale Price by Neighborhood")
        ax.set_xlabel("")
        ax.set_ylabel("Median Price (₹M)")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig); plt.close()

        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(data=df, x="Neighborhood", y="SalePrice", ax=ax, palette="muted")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1e6:.1f}M"))
        ax.set_title("Price Spread by Neighborhood")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig); plt.close()

    with tab3:
        num_cols = ["SalePrice","GrLivArea","TotalBsmtSF","LotArea",
                    "OverallQual","GarageCars","FullBath","Age","TotalSF","Fireplaces"]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig); plt.close()

        top_corr = corr["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)
        st.markdown("#### Top Features Correlated with Price")
        st.bar_chart(top_corr.head(8))

    with tab4:
        feat_x = st.selectbox("X-axis Feature", ["GrLivArea","TotalBsmtSF","LotArea","Age","GarageCars","OverallQual"])
        color_by = st.selectbox("Color By", ["OverallQual","Neighborhood","KitchenQual"])
        fig, ax = plt.subplots(figsize=(8, 5))
        if color_by in ["Neighborhood","KitchenQual"]:
            groups = df[color_by].unique()
            palette = sns.color_palette("tab10", len(groups))
            for g, c in zip(groups, palette):
                sub = df[df[color_by] == g]
                ax.scatter(sub[feat_x], sub["SalePrice"]/1e6, alpha=0.35, s=15, color=c, label=g)
            ax.legend(fontsize=7, ncol=2)
        else:
            sc = ax.scatter(df[feat_x], df["SalePrice"]/1e6, c=df[color_by], cmap="viridis", alpha=0.4, s=15)
            plt.colorbar(sc, ax=ax, label=color_by)
        ax.set_xlabel(feat_x)
        ax.set_ylabel("Sale Price (₹M)")
        ax.set_title(f"Price vs {feat_x}")
        st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<div class="main-header">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare all trained regression models</div>', unsafe_allow_html=True)

    # Metrics table
    st.subheader("📊 Evaluation Metrics (Test Set)")
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "MAE (₹)":  f"₹{m['MAE']:,.0f}",
            "RMSE (₹)": f"₹{m['RMSE']:,.0f}",
            "R² Score": f"{m['R2']:.4f}",
            "Best?": "✅ Best" if name == best_name else ""
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(metrics.keys())
        rmses = [metrics[n]["RMSE"]/1e6 for n in names]
        colors = ["#1f3c88" if n == best_name else "#aab7d4" for n in names]
        ax.barh(names, rmses, color=colors)
        ax.set_xlabel("RMSE (₹M)")
        ax.set_title("RMSE Comparison (lower = better)")
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        r2s = [metrics[n]["R2"] for n in names]
        colors = ["#1f3c88" if n == best_name else "#aab7d4" for n in names]
        ax.barh(names, r2s, color=colors)
        ax.set_xlabel("R² Score")
        ax.set_title("R² Comparison (higher = better)")
        st.pyplot(fig); plt.close()

    # Actual vs Predicted for selected model
    st.subheader("🎯 Actual vs Predicted")
    sel = st.selectbox("Select Model", list(trained_models.keys()), index=list(trained_models.keys()).index(best_name))
    pipe = trained_models[sel]
    NUM, CAT = get_feature_lists()
    pred = pipe.predict(Xte[NUM + CAT])

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(yte/1e6, pred/1e6, alpha=0.35, s=15, color="#4C72B0")
        lims = [min(yte.min(), pred.min())/1e6, max(yte.max(), pred.max())/1e6]
        ax.plot(lims, lims, "r--", lw=2)
        ax.set_xlabel("Actual (₹M)"); ax.set_ylabel("Predicted (₹M)")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig); plt.close()

    with col2:
        residuals = yte - pred
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist(residuals/1e6, bins=40, color="#55a868", edgecolor="white")
        ax.axvline(0, color="red", lw=2, linestyle="--")
        ax.set_xlabel("Residual (₹M)")
        ax.set_title("Residual Distribution")
        st.pyplot(fig); plt.close()

    # Error metrics display
    mae  = mean_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    r2   = r2_score(yte, pred)
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"₹{mae:,.0f}")
    m2.metric("RMSE", f"₹{rmse:,.0f}")
    m3.metric("R²",   f"{r2:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: Feature Insights
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Insights":
    st.markdown('<div class="main-header">🔍 Feature Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understand what drives house prices</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Price by Category", "Age & Renovation"])

    with tab1:
        # Random Forest feature importances
        rf_pipe = trained_models["Random Forest"]
        rf_model = rf_pipe.named_steps["m"]
        pre = rf_pipe.named_steps["pre"]
        NUM, CAT = get_feature_lists()
        try:
            feature_names = NUM.copy()
            ohe = pre.named_transformers_["cat"].named_steps["encoder"]
            cat_names = list(ohe.get_feature_names_out(CAT))
            feature_names += cat_names
            importances = rf_model.feature_importances_
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            # Aggregate back to original names
            orig_imp = {f: 0 for f in NUM + CAT}
            for feat, imp in zip(feature_names, importances):
                matched = False
                for orig in CAT:
                    if feat.startswith(orig + "_"):
                        orig_imp[orig] += imp; matched = True; break
                if not matched and feat in orig_imp:
                    orig_imp[feat] += imp
            imp_series = pd.Series(orig_imp).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            imp_series.plot(kind="barh", ax=ax, color="#1f3c88")
            ax.set_title("Top 15 Feature Importances (Random Forest)")
            ax.set_xlabel("Importance")
            st.pyplot(fig); plt.close()
        except Exception as e:
            st.info("Feature importance chart requires fitted Random Forest model.")

        st.markdown("""
        **Key Drivers of House Price:**
        - 🏗️ **Overall Quality** — biggest single driver
        - 📍 **Neighborhood** — location commands premium  
        - 📐 **Living Area** (GrLivArea) — size matters
        - 🚗 **Garage Cars** — parking adds value
        - 🏚️ **Age** — newer houses are more valuable
        - 🛁 **Bathrooms** — total bath count
        """)

    with tab2:
        cat_feat = st.selectbox("Select Category", ["KitchenQual", "BldgType", "HouseStyle", "MSZoning", "CentralAir", "PavedDrive"])
        fig, ax = plt.subplots(figsize=(9, 5))
        order = df.groupby(cat_feat)["SalePrice"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=cat_feat, y="SalePrice", order=order, ax=ax, palette="muted")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1e6:.1f}M"))
        ax.set_title(f"Sale Price by {cat_feat}")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig); plt.close()

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df["Age"], df["SalePrice"]/1e6, alpha=0.25, s=10, color="#4C72B0")
            z = np.polyfit(df["Age"].dropna(), df.loc[df["Age"].notna(), "SalePrice"]/1e6, 1)
            xs = np.linspace(0, 60, 100)
            ax.plot(xs, np.poly1d(z)(xs), "r--", lw=2)
            ax.set_xlabel("House Age (years)")
            ax.set_ylabel("Price (₹M)")
            ax.set_title("Age vs Price")
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            remo = df[df["WasRemodeled"] == 1]["SalePrice"] / 1e6
            normo = df[df["WasRemodeled"] == 0]["SalePrice"] / 1e6
            ax.hist(normo, bins=30, alpha=0.6, label="Not Remodeled", color="#4C72B0")
            ax.hist(remo,  bins=30, alpha=0.6, label="Remodeled",     color="#DD8452")
            ax.legend()
            ax.set_xlabel("Price (₹M)")
            ax.set_title("Price: Remodeled vs Not")
            st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: Dataset Preview
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Preview":
    st.markdown('<div class="main-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore and filter the raw housing dataset</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        neigh_filter = st.multiselect("Filter Neighborhood", NEIGHBORHOODS, default=NEIGHBORHOODS[:3])
    with col2:
        min_price, max_price = st.slider("Price Range (₹M)", 0.0, 30.0, (0.0, 30.0), 0.5)
    with col3:
        qual_filter = st.slider("Min Quality", 1, 10, 5)

    filtered = df[
        (df["Neighborhood"].isin(neigh_filter if neigh_filter else NEIGHBORHOODS)) &
        (df["SalePrice"] >= min_price * 1e6) &
        (df["SalePrice"] <= max_price * 1e6) &
        (df["OverallQual"] >= qual_filter)
    ]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")
    disp_cols = ["Neighborhood","GrLivArea","OverallQual","YearBuilt","FullBath","GarageCars","SalePrice"]
    display_df = filtered[disp_cols].copy()
    display_df["SalePrice"] = display_df["SalePrice"].apply(lambda v: f"₹{v:,.0f}")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=420)

    st.markdown("#### Dataset Statistics")
    stats2 = filtered[["GrLivArea","TotalBsmtSF","OverallQual","Age","SalePrice"]].describe().round(2)
    st.dataframe(stats2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6: About
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    ## 🏠 House Price Prediction using Regression Models

    This end-to-end machine learning project predicts residential property prices from 
    structural, locational, and quality features. It is designed as a **portfolio project** 
    demonstrating full ML workflow from data to deployment.

    ---

    ### 🎯 Problem Statement
    Manually pricing properties is slow, subjective, and inconsistent. ML-powered Automated 
    Valuation Models (AVMs) allow real estate portals, banks, and investors to generate 
    accurate, instant price estimates at scale.

    ### 🏗️ Architecture
    ```
    Synthetic Data → Preprocessing → Feature Engineering 
        → Regression Models → Evaluation → Streamlit Dashboard
    ```

    ### 🤖 Models Used
    | Model | Type | Notes |
    |---|---|---|
    | Linear Regression | Parametric | Baseline, interpretable |
    | Ridge Regression | Regularized Linear | Handles multicollinearity |
    | Decision Tree | Tree | Non-linear, explainable |
    | Random Forest | Ensemble (Bagging) | Best accuracy + robustness |
    | Gradient Boosting | Ensemble (Boosting) | High accuracy |

    ### 📊 Evaluation Metrics
    - **MAE** (Mean Absolute Error) — average prediction error in ₹
    - **RMSE** (Root Mean Squared Error) — penalizes large errors  
    - **R²** (R-Squared) — % variance explained by the model

    ### 🛠️ Tech Stack
    """)

    tags = ["Python 3.x", "Pandas", "NumPy", "Scikit-learn", "Matplotlib",
            "Seaborn", "Streamlit", "Joblib"]
    st.markdown(" ".join([f'<span class="tag">{t}</span>' for t in tags]), unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🌍 Real-World Applications
    - **Real estate portals** — instant listing valuations
    - **Banks** — mortgage/collateral underwriting
    - **iBuyers** — automated offer generation
    - **Homeowners** — renovation ROI estimation

    ### 📁 Project Structure
    ```
    House-Price-Prediction/
    ├── data/               ← Synthetic housing dataset
    ├── src/                ← Python modules (features, pipeline, training)
    ├── models/             ← Saved trained models
    ├── images/             ← Generated EDA charts
    ├── outputs/            ← Prediction outputs
    ├── notebooks/          ← Jupyter notebooks
    ├── dashboard.py        ← Streamlit dashboard (this app)
    ├── main.py             ← CLI entry point
    ├── requirements.txt    ← Dependencies
    └── README.md           ← Project documentation
    ```
    """)

    st.markdown("---")
    st.markdown("**Built for:** Data Science | ML Engineering | Business Analytics | Data Analyst portfolios")
    st.markdown("**Author:** Student Project · GitHub Portfolio")
