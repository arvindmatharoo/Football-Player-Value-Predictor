# app.py - Enhanced Streamlit app (fixed numeric ranges and unique widget keys)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="Player Value Predictor", layout="wide", initial_sidebar_state="expanded")

# ---- Load pipeline (cached) ----
@st.cache_resource(show_spinner=False)
def load_pipeline(path="value_predictor_pipeline.pkl"):
    return joblib.load(path)

try:
    model_pipeline = load_pipeline()
except Exception as e:
    st.error(f"Could not load model pipeline from 'value_predictor_pipeline.pkl'. Error:\n{e}")
    st.stop()

# ---- Feature lists (must match training) ----
numeric_feats = [
    'age', 'height_cm', 'weight_kgs', 'overall_rating', 'potential', 'wage_euro',
    'crossing','finishing','heading_accuracy','short_passing','volleys','dribbling','curve',
    'freekick_accuracy','long_passing','ball_control','acceleration','sprint_speed','agility',
    'reactions','balance','shot_power','jumping','stamina','strength','long_shots','aggression',
    'interceptions','positioning','vision','penalties','composure','marking','standing_tackle',
    'sliding_tackle'
]
categorical_feats = ['preferred_foot', 'body_type', 'positions', 'nationality']
FEATURES = numeric_feats + categorical_feats

# ---- Extract OHE categories (for selectbox options) ----
def get_ohe_categories(pipe, categorical_features):
    cats = {}
    try:
        preproc = pipe.named_steps['preprocessor']
        ohe = preproc.named_transformers_['cat'].named_steps['onehot']
        ohe_cats = list(ohe.categories_)
        for feat, cat_list in zip(categorical_features, ohe_cats):
            cats[feat] = [str(x) if (pd.notnull(x) and x != '') else 'missing' for x in cat_list]
    except Exception:
        # fallback defaults
        for feat in categorical_features:
            cats[feat] = ['missing']
    return cats

ohe_categories = get_ohe_categories(model_pipeline, categorical_feats)

# ---- Prediction helper (returns euros, confidence, approx std) ----
def predict_and_confidence(df_in):
    missing = [c for c in FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X_in = df_in[FEATURES].copy()
    preds_log = model_pipeline.predict(X_in)
    preds = np.expm1(preds_log)
    # try to compute per-tree std (log-space) -> normalized confidence
    try:
        preproc = model_pipeline.named_steps['preprocessor']
        X_trans = preproc.transform(X_in)
        rf = model_pipeline.named_steps['regressor']
        all_preds = np.stack([est.predict(X_trans) for est in rf.estimators_], axis=0)
        std_log = np.std(all_preds, axis=0)
        std_norm = (std_log - std_log.min()) / (std_log.ptp() + 1e-12)
        confidence = 1.0 - std_norm
        approx_std_euro = np.expm1(preds_log) * std_log
    except Exception:
        confidence = np.ones(len(preds))
        approx_std_euro = np.zeros(len(preds))
    return preds, confidence, approx_std_euro

# ---- Header ----
st.markdown(
    "<div style='background-color:#0b3d91;padding:16px;border-radius:6px'>"
    "<h1 style='color:white;margin:0'>⚽ Player Market Value Predictor</h1>"
    "<p style='color:rgba(255,255,255,0.9);margin:0'>Predict player market value (€) with a trained model.</p>"
    "</div>",
    unsafe_allow_html=True
)
st.write("")

# two-column layout
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Single Player Input")
with col2:
    st.subheader("Batch Upload & Summary")

# ---- Single Player (left) ----
with col1:
    # numeric inputs: two-column layout
    numeric_inputs = {}
    left_col, right_col = st.columns([1,1])
    for i, feat in enumerate(numeric_feats):
        col = left_col if (i % 2 == 0) else right_col

        # sensible defaults and valid ranges
        if feat == 'age':
            val, minv, maxv, step = 21, 15, 50, 1
        elif feat == 'height_cm':
            val, minv, maxv, step = 180, 140, 230, 1
        elif feat == 'weight_kgs':
            val, minv, maxv, step = 75, 40, 140, 1
        elif feat in ['overall_rating', 'potential']:
            val, minv, maxv, step = 70, 1, 99, 1
        elif feat == 'wage_euro':
            val, minv, maxv, step = 5000, 0, 5_000_000, 100
        else:
            # skill attributes typically 0-100
            val, minv, maxv, step = 50, 0, 100, 1

        # unique key for each numeric widget
        widget_key = f"num_{feat}"
        numeric_inputs[feat] = col.number_input(
            label=feat,
            value=int(val),
            min_value=minv,
            max_value=maxv,
            step=step,
            format="%d",
            key=widget_key
        )

    st.write("")  # spacing
    st.write("Categorical attributes")
    cat_cols = st.columns(len(categorical_feats))
    cat_inputs = {}
    for i, feat in enumerate(categorical_feats):
        options = ohe_categories.get(feat, ['missing'])
        if 'missing' not in options:
            options = ['missing'] + options
        widget_key = f"cat_{feat}"
        cat_inputs[feat] = cat_cols[i].selectbox(
            label=feat,
            options=options,
            index=0,
            key=widget_key
        )

    # small image & info
    img_col1, img_col2 = st.columns([1,2])
    with img_col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/7c/Profile_avatar_placeholder_large.png", width=110)
    with img_col2:
        st.markdown("**Preview** — enter attributes and click Predict. The app shows predicted value and confidence.")

    # predict button
    if st.button("Predict Player Value", key="predict_button"):
        input_row = {**numeric_inputs, **cat_inputs}
        input_df = pd.DataFrame([input_row], columns=FEATURES)
        try:
            preds, conf, approx_std = predict_and_confidence(input_df)
            pred_val = float(preds[0])
            conf_pct = float(conf[0]) * 100.0
            st.metric(label="Predicted Market Value", value=f"€{pred_val:,.0f}")
            # colored box
            if pred_val >= 10_000_000:
                box_color = "#BF313B"
            elif pred_val >= 1_000_000:
                box_color = "#9440E8"
            else:
                box_color = "#257B65"
            st.markdown(
                f"<div style='background:{box_color};padding:12px;border-radius:8px'>"
                f"<h2 style='margin:0'>€{pred_val:,.0f}</h2>"
                f"<p style='margin:0;color:#333'>Confidence: {conf_pct:.1f}%</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            if approx_std[0] > 0:
                st.write(f"Approx. predicted std (euros): €{approx_std[0]:,.0f}")
            st.session_state['_last_single_pred'] = float(pred_val)
            st.session_state['_last_single_conf'] = float(conf[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---- Batch Upload & Summary (right) ----
with col2:
    uploaded = st.file_uploader("Upload CSV (must contain model feature columns)", type=["csv"], key="uploader")
    df_batch = None
    if uploaded is not None:
        try:
            df_batch = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_batch = None

        if df_batch is not None:
            missing_cols = [c for c in FEATURES if c not in df_batch.columns]
            if missing_cols:
                st.error(f"Uploaded CSV missing required columns: {missing_cols}")
                df_batch = None
            else:
                preds, confs, approx_std = predict_and_confidence(df_batch)
                df_out = df_batch.copy()
                df_out['predicted_value_euro'] = preds
                df_out['pred_confidence'] = confs
                df_out['pred_std_euro'] = approx_std

                median_pred = float(np.median(preds))
                mean_pred = float(np.mean(preds))
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Batch size", value=f"{len(df_out):,}")
                col_b.metric("Median predicted value", value=f"€{median_pred:,.0f}")
                col_c.metric("Mean predicted value", value=f"€{mean_pred:,.0f}")

                st.write("Predictions (first 10 rows):")
                st.dataframe(df_out.head(10))

                # histogram try plotly, fallback to matplotlib
                try:
                    import plotly.express as px
                    fig = px.histogram(df_out, x='predicted_value_euro', nbins=40, title="Distribution of Predicted Values")
                    if '_last_single_pred' in st.session_state:
                        fig.add_vline(x=st.session_state['_last_single_pred'], line_color='red', line_dash='dash', annotation_text="Single pred", annotation_position="top right")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.write("Histogram of predictions:")
                    st.bar_chart(df_out['predicted_value_euro'].value_counts().nlargest(20))

                csv_bytes = df_out.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", key="download_btn")
    else:
        st.info("Upload a CSV to see batch predictions and summary.")

# ---- Footer ----
st.markdown("---")
with st.container():
    left, right = st.columns([3,1])
    with left:
        st.write("Model info:")
        st.write("- Model type: RandomForestRegressor (pipeline includes preprocessing)")
        st.write("- Example evaluation (MAE): €139,486  — R²: 0.9722")
        st.write("- Predictions invert a log-transform applied to the target during training.")
    with right:
        st.write(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write("Built with Streamlit")
st.write("")
