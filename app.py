"""
Smart Crop Recommendation System - Streamlit Web Application
============================================================
A web interface for predicting optimal crops using trained ML models,
comparing ensemble learning algorithms (Random Forest, Bagging, AdaBoost, Decision Stump),
and exploring unsupervised Soil & Climate K-Means clustering with 2D PCA.
"""

import os
import json
import copy
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.concept_learning import (
    FindS,
    CandidateElimination,
    DEFAULT_ATTRIBUTES,
    DEFAULT_EXAMPLES
)
from src.sequential_covering import (
    SequentialCovering,
    Rule,
    CATEGORICAL_ATTRIBUTES as SC_ATTRIBUTES,
    CATEGORICAL_CROP_DATASET as SC_DATASET
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AgriSmart - Crop Recommendation & Clustering",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Green Agricultural Theme
CUSTOM_CSS = """
<style>
    /* Main Background & Font Styling */
    .stApp {
        background: linear-gradient(135deg, #f8fbf8 0%, #edf7ee 100%);
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Container */
    .hero-header {
        background: linear-gradient(135deg, #1b4d2e 0%, #2e7d32 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(27, 77, 46, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-header p {
        color: #c8e6c9;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Card Containers */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e0ede0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(46, 125, 50, 0.1);
    }

    /* Result Banner */
    .prediction-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #81c784;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.15);
    }
    .crop-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        text-transform: capitalize;
        margin: 0.4rem 0;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #2e7d32;
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .disclaimer-text {
        font-size: 0.82rem;
        color: #4a6b4e;
        margin-top: 0.6rem;
        font-style: italic;
    }

    /* Disclaimer Card for Clustering */
    .clustering-disclaimer {
        background: #fff8e1;
        border-left: 5px solid #ffa000;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0 1.5rem 0;
        color: #5d4037;
        font-size: 0.92rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }

    /* Section Subheadings */
    .section-title {
        color: #1b4d2e;
        font-weight: 700;
        font-size: 1.25rem;
        border-bottom: 2px solid #a5d6a7;
        padding-bottom: 0.3rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #388e3c 0%, #2e7d32 100%);
        box-shadow: 0 6px 14px rgba(46, 125, 50, 0.3);
        transform: translateY(-1px);
    }

    /* Algorithm Comparison Cards */
    .algo-card {
        background: white;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Crop Emoji Dictionary for visual presentation
CROP_ICONS = {
    "rice": "🌾",
    "maize": "🌽",
    "chickpea": "🧆",
    "kidneybeans": "🫘",
    "pigeonpeas": "🌿",
    "mothbeans": "🌱",
    "mungbean": "🫘",
    "blackgram": "🥣",
    "lentil": "🍲",
    "pomegranate": "🪴",
    "banana": "🍌",
    "mango": "🥭",
    "grapes": "🍇",
    "watermelon": "🍉",
    "muskmelon": "🍈",
    "apple": "🍎",
    "orange": "🍊",
    "papaya": "🫐",
    "coconut": "🥥",
    "cotton": "☁️",
    "jute": "🌾",
    "coffee": "☕"
}


# ==========================================
# CACHED LOADERS (AVOID RETRAINING ON RERUN)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_rf_model():
    """
    Load the saved Random Forest model from disk using joblib.
    Uses st.cache_resource to avoid retraining or reloading on every interaction.
    """
    model_path = os.path.join(BASE_DIR, "models", "random_forest.joblib")
    if not os.path.exists(model_path):
        return None, f"Model file not found at '{model_path}'."
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


@st.cache_resource
def load_clustering_models():
    """
    Load StandardScaler, KMeans, and PCA models from disk using joblib.
    Uses st.cache_resource to prevent redundant disk I/O.
    """
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.joblib")
    kmeans_path = os.path.join(BASE_DIR, "models", "kmeans.joblib")
    pca_path = os.path.join(BASE_DIR, "models", "pca.joblib")

    if not (os.path.exists(scaler_path) and os.path.exists(kmeans_path) and os.path.exists(pca_path)):
        return None, None, None, "Clustering models missing. Please run 'python src/clustering.py'."
    try:
        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)
        pca = joblib.load(pca_path)
        return scaler, kmeans, pca, None
    except Exception as e:
        return None, None, None, f"Error loading clustering models: {str(e)}"


@st.cache_data
def load_metadata():
    """
    Load features, classes, and model metrics from the models/ directory.
    """
    features_path = os.path.join(BASE_DIR, "models", "features.json")
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    classes_path = os.path.join(BASE_DIR, "models", "classes.json")

    features = None
    classes = None
    metrics_data = None
    errors = []

    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            features = json.load(f)
    else:
        errors.append(f"Missing {features_path}")

    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            classes = json.load(f)

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
    else:
        errors.append(f"Missing {metrics_path}")

    return features, classes, metrics_data, errors


@st.cache_data
def load_clustering_data():
    """
    Load clustering metrics JSON and clustered CSV dataset.
    """
    metrics_path = os.path.join(BASE_DIR, "models", "clustering_metrics.json")
    csv_path = os.path.join(BASE_DIR, "data", "clustered_crops.csv")

    if not (os.path.exists(metrics_path) and os.path.exists(csv_path)):
        return None, None, "Clustering metrics or clustered dataset missing. Please run 'python src/clustering.py'."
    try:
        with open(metrics_path, "r") as f:
            c_metrics = json.load(f)
        df_clustered = pd.read_csv(csv_path)
        return c_metrics, df_clustered, None
    except Exception as e:
        return None, None, f"Error loading clustering data: {str(e)}"



# Top Hero Header
st.markdown("""
<div class="hero-header">
    <h1>🌱 AgriSmart: Intelligent Crop Recommendation</h1>
    <p>Empowering precision agriculture with Machine Learning ensemble algorithms, K-Means clustering, and soil-climate analytics.</p>
</div>
""", unsafe_allow_html=True)

# Load Supervised Model & Metadata
rf_model, model_err = load_rf_model()
features, classes, metrics_data, meta_errors = load_metadata()

# Display error if supervised models are missing
if model_err or meta_errors:
    st.error("⚠️ **Supervised Model Artifacts Missing or Incomplete**")
    st.markdown("""
    The trained models or evaluation metrics were not found in the `models/` directory.
    
    Please run the training script first:
    ```bash
    python src/train.py
    ```
    Once training completes, reload this page.
    """)
    if model_err:
        st.info(f"Model status: {model_err}")
    if meta_errors:
        st.info(f"Metadata status: {', '.join(meta_errors)}")
    st.stop()


# Sidebar Navigation & Information
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.title("🌾 AgriSmart Navigator")
    nav_choice = st.radio(
        "Select Application View:",
        [
            "🚜 Crop Recommendation",
            "📊 Model Comparison & Benchmarks",
            "🗺️ Soil & Climate Clustering",
            "🧪 Concept Learning Lab",
            "📜 Sequential Covering (Rule Learning)",
            "🧠 Algorithm Explanations"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("ℹ️ System Status")
    st.markdown(f"**Supervised Model**: `Random Forest`")
    if metrics_data and "models" in metrics_data and "Random Forest" in metrics_data["models"]:
        rf_acc = metrics_data["models"]["Random Forest"]["accuracy"] * 100
        st.markdown(f"**RF Test Accuracy**: `{rf_acc:.2f}%`")
    st.markdown(f"**Clustering**: `K-Means (k=4)`")
    st.markdown(f"**Crops Supported**: `{len(classes) if classes else 22}` varieties")
    st.markdown(f"**Input Parameters**: `7` Soil & Climate features")
    st.caption("AgriSmart ML-II Project • Python & Scikit-learn")


# ==========================================
# VIEW 1: CROP RECOMMENDATION
# ==========================================
if nav_choice == "🚜 Crop Recommendation":
    st.markdown('<div class="section-title">🌿 Enter Soil Nutrients & Environmental Parameters</div>', unsafe_allow_html=True)
    st.write("Provide the farm's soil analysis and current meteorological conditions to receive real-time crop recommendations.")

    # Quick Presets for User Convenience
    preset_col1, preset_col2, preset_col3, preset_col4, preset_col5 = st.columns([1, 1, 1, 1, 1.2])
    with preset_col1:
        st.write("**Quick Presets:**")
    with preset_col2:
        if st.button("🌾 Rice Farm", help="Preset: High rainfall & humidity"):
            st.session_state["n_val"] = 90.0
            st.session_state["p_val"] = 42.0
            st.session_state["k_val"] = 43.0
            st.session_state["temp_val"] = 21.0
            st.session_state["hum_val"] = 82.0
            st.session_state["ph_val"] = 6.5
            st.session_state["rain_val"] = 202.0
    with preset_col3:
        if st.button("🍎 Apple Orchard", help="Preset: High Potassium & cooler temperature"):
            st.session_state["n_val"] = 24.0
            st.session_state["p_val"] = 128.0
            st.session_state["k_val"] = 196.0
            st.session_state["temp_val"] = 22.8
            st.session_state["hum_val"] = 90.7
            st.session_state["ph_val"] = 5.5
            st.session_state["rain_val"] = 110.4
    with preset_col4:
        if st.button("☕ Coffee Plantation", help="Preset: High Nitrogen & tropical humidity"):
            st.session_state["n_val"] = 91.0
            st.session_state["p_val"] = 21.0
            st.session_state["k_val"] = 26.0
            st.session_state["temp_val"] = 26.3
            st.session_state["hum_val"] = 57.4
            st.session_state["ph_val"] = 7.3
            st.session_state["rain_val"] = 191.7
    with preset_col5:
        if st.button("🔄 Reset Defaults"):
            st.session_state["n_val"] = 50.0
            st.session_state["p_val"] = 50.0
            st.session_state["k_val"] = 50.0
            st.session_state["temp_val"] = 25.0
            st.session_state["hum_val"] = 70.0
            st.session_state["ph_val"] = 6.5
            st.session_state["rain_val"] = 100.0

    # Initialize session state if not set
    default_vals = {
        "n_val": 50.0,
        "p_val": 50.0,
        "k_val": 50.0,
        "temp_val": 25.0,
        "hum_val": 70.0,
        "ph_val": 6.5,
        "rain_val": 100.0
    }
    for k, v in default_vals.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Input Form Layout in 3 Columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🧪 Soil Macro-Nutrients")
        n_input = st.number_input(
            "Nitrogen (N) ratio in soil",
            min_value=0.0,
            max_value=200.0,
            value=float(st.session_state["n_val"]),
            step=1.0,
            help="Nitrogen content in soil (kg/ha). Essential for leaf growth and chlorophyll."
        )
        p_input = st.number_input(
            "Phosphorus (P) ratio in soil",
            min_value=0.0,
            max_value=200.0,
            value=float(st.session_state["p_val"]),
            step=1.0,
            help="Phosphorus content in soil (kg/ha). Essential for root development and flowering."
        )
        k_input = st.number_input(
            "Potassium (K) ratio in soil",
            min_value=0.0,
            max_value=250.0,
            value=float(st.session_state["k_val"]),
            step=1.0,
            help="Potassium content in soil (kg/ha). Regulates water balance and disease resistance."
        )

    with col2:
        st.markdown("#### ⛅ Climate & Weather")
        temp_input = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=55.0,
            value=float(st.session_state["temp_val"]),
            step=0.5,
            help="Average ambient temperature in degrees Celsius."
        )
        hum_input = st.number_input(
            "Relative Humidity (%)",
            min_value=5.0,
            max_value=100.0,
            value=float(st.session_state["hum_val"]),
            step=1.0,
            help="Relative air humidity percentage (5% to 100%)."
        )
        rain_input = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=float(st.session_state["rain_val"]),
            step=5.0,
            help="Annual or seasonal rainfall in millimeters."
        )

    with col3:
        st.markdown("#### 🪨 Soil Chemistry")
        ph_input = st.number_input(
            "Soil pH Level (0 - 14)",
            min_value=0.0,
            max_value=14.0,
            value=float(st.session_state["ph_val"]),
            step=0.1,
            help="Soil acidity/alkalinity scale. Optimal range for most crops is 5.5 to 7.5."
        )
        st.markdown("""
        <div style="background:#f1f8f2; border-radius:8px; padding:12px; margin-top:20px; font-size:0.88rem; color:#2e5b32;">
            <b>💡 Agronomy Tip:</b><br>
            • pH &lt; 5.5: Strongly Acidic<br>
            • pH 6.0 – 7.0: Optimal for most crops<br>
            • pH &gt; 8.0: Alkaline / Calcareous soil
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    recommend_button = st.button("🌱 Recommend Optimal Crop", use_container_width=True)

    if recommend_button:
        # Input Validation
        validation_errors = []
        if ph_input < 3.0 or ph_input > 10.0:
            validation_errors.append("Soil pH is outside normal agricultural ranges (3.0 to 10.0).")
        if hum_input <= 0 or hum_input > 100:
            validation_errors.append("Humidity must be between 1% and 100%.")

        if validation_errors:
            for err in validation_errors:
                st.warning(f"⚠️ {err}")

        # Assemble features into exact order expected by model
        input_data = pd.DataFrame([{
            "N": n_input,
            "P": p_input,
            "K": k_input,
            "temperature": temp_input,
            "humidity": hum_input,
            "ph": ph_input,
            "rainfall": rain_input
        }])[features]

        with st.spinner("Analyzing soil and climatic patterns using Random Forest..."):
            predicted_crop = rf_model.predict(input_data)[0]

            # Model Confidence Calculation
            has_proba = hasattr(rf_model, "predict_proba")
            confidence_str = ""
            top3_crops = []

            if has_proba:
                probabilities = rf_model.predict_proba(input_data)[0]
                pred_idx = np.where(rf_model.classes_ == predicted_crop)[0][0]
                confidence_score = probabilities[pred_idx] * 100
                confidence_str = f"{confidence_score:.1f}%"

                # Sort top 3 alternatives
                top_indices = np.argsort(probabilities)[::-1][:3]
                top3_crops = [(rf_model.classes_[i], probabilities[i] * 100) for i in top_indices]

        # Display Result
        crop_icon = CROP_ICONS.get(predicted_crop.lower(), "🌱")
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size: 1.1rem; color: #2e7d32; font-weight: 600;">RECOMMENDED CROP MATCH</div>
            <div class="crop-name">{crop_icon} {predicted_crop.upper()}</div>
            {f'<div class="confidence-badge">Model Confidence: {confidence_str}</div>' if has_proba else ''}
            <div class="disclaimer-text">
                📌 Note: Model confidence indicates statistical pattern certainty learned from training data. It is not an agronomic guarantee of crop yield or field outcome. Always verify with local soil testing and agricultural extension guidelines.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show Top 3 Crop Probability Distribution
        if top3_crops and len(top3_crops) > 1:
            st.markdown("#### 🌾 Top Alternative Crop Candidates")
            prob_cols = st.columns(len(top3_crops))
            for i, (alt_crop, alt_prob) in enumerate(top3_crops):
                with prob_cols[i]:
                    alt_icon = CROP_ICONS.get(alt_crop.lower(), "🌱")
                    st.metric(
                        label=f"Rank {i+1}: {alt_icon} {alt_crop.capitalize()}",
                        value=f"{alt_prob:.1f}%",
                        delta=f"Rank {i+1}"
                    )
                    st.progress(float(alt_prob / 100.0))


# ==========================================
# VIEW 2: MODEL COMPARISON & BENCHMARKS
# ==========================================
elif nav_choice == "📊 Model Comparison & Benchmarks":
    st.markdown('<div class="section-title">📊 ML-II Model Performance & Benchmarking</div>', unsafe_allow_html=True)
    st.write("All metrics below are dynamically loaded from `models/model_metrics.json` recorded on the 20% holdout test set (440 samples).")

    if not metrics_data or "models" not in metrics_data:
        st.warning("No metrics data found in `models/model_metrics.json`.")
    else:
        models_dict = metrics_data["models"]

        # 1. Summary Metrics Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        for idx, (m_name, m_info) in enumerate(models_dict.items()):
            col = [col_m1, col_m2, col_m3, col_m4][idx % 4]
            with col:
                acc_pct = m_info["accuracy"] * 100
                f1_val = m_info["f1_weighted"]
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem; color:#555; font-weight:600;">{m_name}</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#1b5e20;">{acc_pct:.2f}%</div>
                    <div style="font-size:0.82rem; color:#777;">F1: {f1_val:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        # 2. Performance Comparison Table
        st.markdown("#### 📋 Detailed Metrics Summary Table")
        table_rows = []
        for name, data in models_dict.items():
            table_rows.append({
                "Model Algorithm": name,
                "Accuracy (%)": f"{data['accuracy'] * 100:.2f}%",
                "Weighted Precision": f"{data['precision_weighted']:.4f}",
                "Weighted Recall": f"{data['recall_weighted']:.4f}",
                "Weighted F1-Score": f"{data['f1_weighted']:.4f}",
                "Model File": data.get("filename", "N/A")
            })
        comparison_df = pd.DataFrame(table_rows)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # 3. Visual Comparison Chart
        st.markdown("#### 📈 Multi-Metric Comparative Chart")
        chart_data = []
        for name, data in models_dict.items():
            chart_data.append({
                "Model": name,
                "Accuracy": data["accuracy"],
                "Precision": data["precision_weighted"],
                "Recall": data["recall_weighted"],
                "F1-Score": data["f1_weighted"]
            })
        plot_df = pd.DataFrame(chart_data)

        # Matplotlib / Seaborn visualization
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        plot_melted = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

        palette = ["#2e7d32", "#66bb6a", "#81c784", "#a5d6a7"]
        sns.barplot(
            data=plot_melted,
            x="Model",
            y="Score",
            hue="Metric",
            palette=palette,
            ax=ax
        )
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold", color="#1b4d2e")
        ax.set_xlabel("Machine Learning Algorithm", fontsize=11, fontweight="bold", color="#1b4d2e")
        ax.set_title("ML-II Test Set Performance Across 4 Evaluation Metrics", fontsize=12, fontweight="bold", pad=12, color="#1b4d2e")
        ax.legend(title="Metric", loc="upper left", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Value labels on bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0.03:
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=8, color="#222",
                            xytext=(0, 2), textcoords='offset points')

        plt.tight_layout()
        st.pyplot(fig)

        # 4. Key Takeaways & Technical Insights
        st.markdown("#### 💡 Key Takeaways from Empirical Evaluation")
        st.markdown("""
        - **Random Forest (99.55%)**: Best overall classifier. Combining 100 de-correlated decision trees with random feature subsampling eliminates individual tree variance and maps high-dimensional non-linear crop boundaries with near-perfect accuracy.
        - **Bagging Classifier (99.09%)**: Boosts stability by training 100 unconstrained decision trees on independent bootstrap samples. Demonstrates that ensemble variance reduction works exceptionally well on agricultural tabular data.
        - **AdaBoost Classifier (31.36%)**: Boosting weak learners (stumps) sequentially improves from 9.09% to 31.36%, but 100 single-split stumps remain underfit for a 22-class non-linear multi-class problem.
        - **Decision Stump (9.09%)**: A single threshold split (`max_depth=1`) can only partition data into 2 branches, making it mathematically incapable of directly distinguishing 22 crop classes, illustrating the classic definition of a weak learner.
        """)


# ==========================================
# VIEW 3: SOIL & CLIMATE CLUSTERING (NEW)
# ==========================================
elif nav_choice == "🗺️ Soil & Climate Clustering":
    st.markdown('<div class="section-title">🗺️ Unsupervised Soil & Climate Clustering (K-Means)</div>', unsafe_allow_html=True)
    st.write("Discover natural data-driven groupings across the 7 soil and climate dimensions without using crop labels.")

    # Load Clustering Models & Data
    scaler, kmeans, pca, cluster_model_err = load_clustering_models()
    c_metrics, df_clustered, cluster_data_err = load_clustering_data()

    if cluster_model_err or cluster_data_err:
        st.error("⚠️ **Clustering Artifacts Missing**")
        st.markdown("""
        Clustering models or data files were not found in `models/` or `data/`.
        
        Please run the clustering script first:
        ```bash
        python src/clustering.py
        ```
        Once the clustering script finishes, reload this page.
        """)
        if cluster_model_err:
            st.info(f"Model status: {cluster_model_err}")
        if cluster_data_err:
            st.info(f"Data status: {cluster_data_err}")
    else:
        # Mandatory Agronomic Disclaimer (Requirement 10)
        st.markdown("""
        <div class="clustering-disclaimer">
            <b>⚠️ Important Agronomic Interpretation Notice:</b><br>
            These four clusters are <b>unsupervised, data-driven statistical patterns</b> discovered through K-Means clustering on the standardized numerical features (N, P, K, temperature, humidity, pH, and rainfall). 
            <b>They are NOT officially verified agricultural suitability zones or agronomic soil classes.</b> 
            Real-world farm cultivation requires localized soil tests, micro-climate verification, and regional agronomic advisories.
        </div>
        """, unsafe_allow_html=True)

        # 1. Summary Cards for Clustering
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.88rem; color:#555; font-weight:600;">Number of Clusters (k)</div>
                <div style="font-size:1.8rem; font-weight:800; color:#1b5e20;">{c_metrics.get('n_clusters', 4)}</div>
                <div style="font-size:0.8rem; color:#777;">K-Means (k=4, random_state=42)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            sil_score = c_metrics.get("silhouette_score", 0.2553)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.88rem; color:#555; font-weight:600;">Silhouette Score</div>
                <div style="font-size:1.8rem; font-weight:800; color:#2e7d32;">{sil_score:.4f}</div>
                <div style="font-size:0.8rem; color:#777;">Cluster cohesion & separation</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c3:
            total_samples = len(df_clustered)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.88rem; color:#555; font-weight:600;">Total Dataset Samples</div>
                <div style="font-size:1.8rem; font-weight:800; color:#1b5e20;">{total_samples:,}</div>
                <div style="font-size:0.8rem; color:#777;">StandardScaler normalized</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c4:
            exp_var = c_metrics.get("pca_explained_variance_ratio", [0.276, 0.185])
            total_var = sum(exp_var) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.88rem; color:#555; font-weight:600;">PCA 2D Variance Captured</div>
                <div style="font-size:1.8rem; font-weight:800; color:#2e7d32;">{total_var:.2f}%</div>
                <div style="font-size:0.8rem; color:#777;">PC1: {exp_var[0]*100:.1f}%, PC2: {exp_var[1]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Charts Section: Cluster Size & 2D PCA Scatter Plot
        chart_col1, chart_col2 = st.columns(2)

        cluster_palette = ["#2e7d32", "#1976d2", "#f57c00", "#7b1fa2"]
        cluster_names = [f"Cluster {i}" for i in range(4)]

        with chart_col1:
            st.markdown("#### 📊 Cluster Size Distribution")
            sizes_dict = c_metrics.get("cluster_sizes", {})
            size_df = pd.DataFrame([
                {"Cluster": f"Cluster {k}", "Samples": v, "Percentage": f"{(v/total_samples)*100:.1f}%"}
                for k, v in sizes_dict.items()
            ])

            fig_size, ax_size = plt.subplots(figsize=(6, 4.2), dpi=150)
            bars = ax_size.bar(
                size_df["Cluster"],
                size_df["Samples"],
                color=cluster_palette[:len(size_df)],
                edgecolor="#ffffff",
                linewidth=1.5
            )
            ax_size.set_ylabel("Number of Samples", fontsize=10, fontweight="bold", color="#1b4d2e")
            ax_size.set_title("Distribution of 2,200 Agricultural Profiles Across 4 Clusters", fontsize=10, fontweight="bold", pad=10, color="#1b4d2e")
            ax_size.grid(axis="y", linestyle="--", alpha=0.5)

            # Annotate values
            for bar in bars:
                height = bar.get_height()
                ax_size.annotate(f"{height}\n({(height/total_samples)*100:.1f}%)",
                                (bar.get_x() + bar.get_width() / 2., height),
                                ha='center', va='bottom',
                                fontsize=8.5, fontweight="bold", color="#333",
                                xytext=(0, 2), textcoords='offset points')
            ax_size.set_ylim(0, max(sizes_dict.values()) * 1.2)
            plt.tight_layout()
            st.pyplot(fig_size)

        with chart_col2:
            st.markdown("#### 🔍 2D PCA Cluster Projection")
            fig_pca, ax_pca = plt.subplots(figsize=(6, 4.2), dpi=150)

            # Scatter plot of samples
            for c_id in sorted(df_clustered["cluster"].unique()):
                sub = df_clustered[df_clustered["cluster"] == c_id]
                ax_pca.scatter(
                    sub["pca_1"],
                    sub["pca_2"],
                    c=cluster_palette[c_id % len(cluster_palette)],
                    label=f"Cluster {c_id} ({len(sub)} samples)",
                    alpha=0.6,
                    edgecolor="none",
                    s=28
                )

            # Project KMeans cluster centers through PCA for visual landmarks
            centers_scaled = kmeans.cluster_centers_
            centers_pca = pca.transform(centers_scaled)
            ax_pca.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                c="black",
                marker="X",
                s=110,
                linewidths=1.5,
                edgecolors="white",
                label="Cluster Centroids",
                zorder=5
            )

            ax_pca.set_xlabel(f"PCA Component 1 ({exp_var[0]*100:.2f}% var)", fontsize=9, fontweight="bold", color="#1b4d2e")
            ax_pca.set_ylabel(f"PCA Component 2 ({exp_var[1]*100:.2f}% var)", fontsize=9, fontweight="bold", color="#1b4d2e")
            ax_pca.set_title("7D Soil/Climate Features Reduced to 2D via PCA", fontsize=10, fontweight="bold", pad=10, color="#1b4d2e")
            ax_pca.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.9)
            ax_pca.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_pca)

        # 3. Cluster Profiles in Original Physical Units
        st.markdown("#### 📋 Cluster Profiles (Mean Values in Original Physical Units)")
        st.write("Compare the average soil nutrients and environmental conditions for each cluster in real-world units (kg/ha, °C, %, mm, pH):")

        profiles_dict = c_metrics.get("cluster_profiles_original_units", {})
        profile_rows = []
        for c_id, vals in profiles_dict.items():
            top_crops_list = c_metrics.get("top_crops_per_cluster", {}).get(str(c_id), [])
            top_crops_str = ", ".join([f"{item['crop'].capitalize()}" for item in top_crops_list[:4]])
            row = {
                "Cluster": f"Cluster {c_id}",
                "Samples": sizes_dict.get(str(c_id), 0),
                "Nitrogen (N)": f"{vals['N']:.1f} kg/ha",
                "Phosphorus (P)": f"{vals['P']:.1f} kg/ha",
                "Potassium (K)": f"{vals['K']:.1f} kg/ha",
                "Temp (°C)": f"{vals['temperature']:.1f} °C",
                "Humidity (%)": f"{vals['humidity']:.1f}%",
                "Soil pH": f"{vals['ph']:.2f}",
                "Rainfall (mm)": f"{vals['rainfall']:.1f} mm",
                "Frequent Crops Found": top_crops_str
            }
            profile_rows.append(row)

        profile_df = pd.DataFrame(profile_rows)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

        # 4. Detailed Narrative Cards for Each Cluster
        st.markdown("#### 🌾 Agronomic Characteristics of the 4 Discovered Clusters")
        cluster_cards = [
            {
                "id": 0,
                "title": "Cluster 0: Legume & Pulse Moderate-Nutrient Zone",
                "color": "#2e7d32",
                "summary": "Characterized by lower nitrogen (~26 kg/ha), moderate phosphorus (~57 kg/ha), and lower relative humidity (~50%).",
                "crops": "Chickpea (100%), Kidneybeans (100%), Mothbeans (100%), Blackgram (100%), Lentil (100%).",
                "takeaway": "These crops are nitrogen-fixing legumes or drought-tolerant pulses that thrive in lower-humidity soil environments without heavy synthetic nitrogen fertilization."
            },
            {
                "id": 1,
                "title": "Cluster 1: High-Rainfall Tropical & Moisture Zone",
                "color": "#1976d2",
                "summary": "Characterized by the highest rainfall (~165 mm), high humidity (~85.6%), and balanced N-P-K nutrient ratios.",
                "crops": "Rice (100%), Pomegranate (100%), Orange (100%), Coconut (100%), Jute (99%).",
                "takeaway": "Groups moisture-intensive crops and tropical plantations requiring abundant precipitation and continuous atmospheric humidity."
            },
            {
                "id": 2,
                "title": "Cluster 2: Heavy Nitrogen Feeders & Warm Climate Zone",
                "color": "#f57c00",
                "summary": "Characterized by very high nitrogen demand (~96.6 kg/ha), high humidity (~80%), and moderate rainfall (~75 mm).",
                "crops": "Banana (100%), Watermelon (100%), Muskmelon (100%), Cotton (100%), Maize (82%).",
                "takeaway": "Captures fast-growing, nutrient-hungry cash crops that require high nitrogen inputs for vigorous vegetative and fruit expansion."
            },
            {
                "id": 3,
                "title": "Cluster 3: Extreme Potassium-Rich Fruit Orchard Zone",
                "color": "#7b1fa2",
                "summary": "Characterized by extraordinarily high potassium (~200 kg/ha) and phosphorus (~133 kg/ha) with lower soil pH (~5.98).",
                "crops": "Grapes (100%), Apple (100%).",
                "takeaway": "Clusters all 100 apple and 100 grape records together. Both commercial fruits demand high potassium for sugar accumulation and cold/temperate orchard cultivation."
            }
        ]

        for card in cluster_cards:
            st.markdown(f"""
            <div style="background:white; border-left: 6px solid {card['color']}; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.9rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="color:{card['color']}; margin:0;">{card['title']}</h4>
                    <span style="background:#f5f5f5; color:#444; font-size:0.85rem; font-weight:700; padding:3px 10px; border-radius:12px;">
                        {sizes_dict.get(str(card['id']), 0)} samples
                    </span>
                </div>
                <div style="color:#333; margin-top:8px; font-size:0.93rem;"><b>Environmental Profile:</b> {card['summary']}</div>
                <div style="color:#1b5e20; margin-top:4px; font-size:0.93rem;"><b>Dominant Crops Found:</b> {card['crops']}</div>
                <div style="color:#666; margin-top:6px; font-size:0.88rem; font-style:italic;">💡 <b>Observation:</b> {card['takeaway']}</div>
            </div>
            """, unsafe_allow_html=True)

        # 5. Interactive Farm Cluster Classifier
        st.markdown("#### 🧪 Test Your Farm's Cluster Membership")
        st.write("Using the current inputs from the Crop Recommendation panel, see which unsupervised data-driven cluster your soil and climate map to:")

        test_col1, test_col2 = st.columns([1, 2])
        with test_col1:
            if st.button("📍 Identify Farm Cluster for Current Input", use_container_width=True):
                # Retrieve current session values
                curr_input = np.array([[
                    st.session_state.get("n_val", 50.0),
                    st.session_state.get("p_val", 50.0),
                    st.session_state.get("k_val", 50.0),
                    st.session_state.get("temp_val", 25.0),
                    st.session_state.get("hum_val", 70.0),
                    st.session_state.get("ph_val", 6.5),
                    st.session_state.get("rain_val", 100.0)
                ]])
                
                # Standardize using fitted scaler
                curr_scaled = scaler.transform(curr_input)
                assigned_cluster = int(kmeans.predict(curr_scaled)[0])
                curr_pca = pca.transform(curr_scaled)[0]

                card_match = cluster_cards[assigned_cluster]
                st.session_state["assigned_cluster_info"] = {
                    "cluster": assigned_cluster,
                    "title": card_match["title"],
                    "color": card_match["color"],
                    "crops": card_match["crops"],
                    "pca_1": curr_pca[0],
                    "pca_2": curr_pca[1]
                }

        with test_col2:
            if "assigned_cluster_info" in st.session_state:
                c_info = st.session_state["assigned_cluster_info"]
                st.markdown(f"""
                <div style="background:#e8f5e9; border: 2px solid {c_info['color']}; border-radius: 12px; padding: 12px 18px;">
                    <div style="font-weight:700; color:{c_info['color']}; font-size:1.1rem;">Mapped to: {c_info['title']}</div>
                    <div style="font-size:0.9rem; color:#222; margin-top:4px;">
                        <b>Representative Crops in Pattern:</b> {c_info['crops']}
                    </div>
                    <div style="font-size:0.8rem; color:#666; margin-top:4px;">
                        PCA 2D Coordinates: (PC1: {c_info['pca_1']:.2f}, PC2: {c_info['pca_2']:.2f})
                    </div>
                </div>
                """, unsafe_allow_html=True)


# =====================================================================
# VIEW 4: CONCEPT LEARNING LAB (FIND-S & CANDIDATE ELIMINATION)
# =====================================================================
elif nav_choice == "🧪 Concept Learning Lab":
    st.markdown('<div class="section-title">🧪 Concept Learning Lab (Find-S & Candidate Elimination)</div>', unsafe_allow_html=True)
    st.write("Explore classical symbolic Machine Learning algorithms that search discrete hypothesis spaces to learn a target concept from labeled categorical training instances.")

    # Educational Disclaimer (Requirement 8 & 12)
    st.markdown("""
    <div class="clustering-disclaimer" style="border-left-color: #0288d1; background: #e1f5fe; color: #01579b;">
        <b>🎓 Educational Machine Learning Demonstration Notice:</b><br>
        Find-S and Candidate Elimination are foundational symbolic algorithms from classical ML theory (Tom Mitchell, <i>Machine Learning</i>, Ch. 2).<br>
        <b>They are pedagogical demonstrations using categorical teaching attributes and do NOT predict real-world agricultural crop suitability.</b><br>
        • <b>Find-S</b>: Learns the maximally specific hypothesis consistent with positive examples and <u>completely ignores all negative training examples</u>.<br>
        • <b>Candidate Elimination</b>: Maintains the dual boundaries of the Version Space ($S$ and $G$) and <u>strictly assumes a consistent, noise-free dataset</u>.
    </div>
    """, unsafe_allow_html=True)

    # Target Concept Definition
    st.markdown("#### 🎯 Target Concept: Wetland Rice Cultivation Suitability")
    st.markdown("""
    - **Target Concept $c(x)$**: `Suitable for High-Moisture Wetland Rice Cultivation (Yes / No)`
    - **Hypothesis Representation**: Conjunction of constraints $\\langle \\text{SoilType}, \\text{Climate}, \\text{Moisture}, \\text{Rainfall} \\rangle$ where each attribute is a specific value, `?` (any value accepted), or `0` (no value accepted).
    """)

    # Dataset Selection & Configuration
    dataset_mode = st.radio(
        "Choose Training Dataset Scenario:",
        [
            "🌾 Standard Wetland Rice Dataset (Consistent, 5 Examples)",
            "⚡ Noisy / Contradictory Dataset (Demonstrates Version Space Collapse)",
            "✏️ Custom Interactive Dataset"
        ],
        index=0,
        horizontal=True
    )

    if dataset_mode == "🌾 Standard Wetland Rice Dataset (Consistent, 5 Examples)":
        active_examples = DEFAULT_EXAMPLES
        active_attributes = DEFAULT_ATTRIBUTES
    elif dataset_mode == "⚡ Noisy / Contradictory Dataset (Demonstrates Version Space Collapse)":
        # Introduce a direct contradiction: Example 1 is labeled Yes, but Example 6 with identical features is labeled No!
        active_attributes = DEFAULT_ATTRIBUTES
        active_examples = [
            {
                "id": 1,
                "features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Heavy"},
                "label": "Yes",
                "rationale": "Labeled positive by agricultural surveyor A."
            },
            {
                "id": 2,
                "features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Moderate"},
                "label": "Yes",
                "rationale": "Standard positive conditions."
            },
            {
                "id": 3,
                "features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Heavy"},
                "label": "No",
                "rationale": "CONTRADICTION: Exactly identical features to Example 1, but labeled Negative due to data entry noise!"
            }
        ]
        st.warning("⚠️ **Simulating Noisy Data**: Example 3 has identical attributes to Example 1 but opposite label ('No'). Observe how Candidate Elimination handles this.")
    else:
        # Custom Interactive Dataset
        active_attributes = DEFAULT_ATTRIBUTES
        if "custom_cl_examples" not in st.session_state:
            st.session_state["custom_cl_examples"] = copy.deepcopy(DEFAULT_EXAMPLES)

        st.markdown("##### ➕ Add a Custom Training Example")
        c_col1, c_col2, c_col3, c_col4, c_col5 = st.columns([1.5, 1.5, 1.5, 1.5, 1.2])
        with c_col1:
            c_soil = st.selectbox("SoilType", DEFAULT_ATTRIBUTES["SoilType"], key="cl_soil")
        with c_col2:
            c_clim = st.selectbox("Climate", DEFAULT_ATTRIBUTES["Climate"], key="cl_clim")
        with c_col3:
            c_moist = st.selectbox("Moisture", DEFAULT_ATTRIBUTES["Moisture"], key="cl_moist")
        with c_col4:
            c_rain = st.selectbox("Rainfall", DEFAULT_ATTRIBUTES["Rainfall"], key="cl_rain")
        with c_col5:
            c_lbl = st.selectbox("Label", ["Yes", "No"], key="cl_lbl")

        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("➕ Add Example to Dataset", use_container_width=True):
                new_id = len(st.session_state["custom_cl_examples"]) + 1
                st.session_state["custom_cl_examples"].append({
                    "id": new_id,
                    "features": {"SoilType": c_soil, "Climate": c_clim, "Moisture": c_moist, "Rainfall": c_rain},
                    "label": c_lbl,
                    "rationale": f"User-added custom training example #{new_id}"
                })
                st.rerun()
        with btn_c2:
            if st.button("🔄 Reset Custom Dataset", use_container_width=True):
                st.session_state["custom_cl_examples"] = copy.deepcopy(DEFAULT_EXAMPLES)
                st.rerun()

        active_examples = st.session_state["custom_cl_examples"]

    # 1. Display Current Training Examples Table
    st.markdown("#### 📋 Training Examples in Current Dataset")
    ex_table = []
    for ex in active_examples:
        ex_table.append({
            "Example ID": f"#{ex['id']}",
            "Soil Type": ex["features"]["SoilType"],
            "Climate": ex["features"]["Climate"],
            "Moisture": ex["features"]["Moisture"],
            "Rainfall": ex["features"]["Rainfall"],
            "Label": "✅ Yes (+)" if ex["label"] in ["Yes", "+"] else "❌ No (-)",
            "Rationale": ex.get("rationale", "")
        })
    st.dataframe(pd.DataFrame(ex_table), use_container_width=True, hide_index=True)

    # 2. Run Both Algorithms
    find_s = FindS(list(active_attributes.keys()))
    final_h = find_s.fit(active_examples)

    ce = CandidateElimination(active_attributes)
    final_s, final_g = ce.fit(active_examples)

    # 3. Final Boundary Summary Cards
    st.markdown("#### 🏁 Final Learned Concept Boundaries")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid #2e7d32;">
            <div style="font-weight:700; color:#1b5e20; font-size:1.15rem;">Find-S Final Hypothesis</div>
            <div style="font-size:1.45rem; font-weight:800; color:#2e7d32; margin:10px 0; font-family:monospace;">
                ⟨{', '.join(final_h)}⟩
            </div>
            <div style="font-size:0.85rem; color:#444; line-height:1.5;">
                • <b>Type:</b> Maximally Specific Hypothesis ($h_{{find-s}}$)<br>
                • <b>Behavior:</b> Strictly generalizes on positive examples ($+$).<br>
                • <b>Limitation:</b> <u>Completely ignores negative examples</u> ($-$), so it cannot detect contradictory data or bound uncertainty.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        if ce.is_version_space_empty:
            st.markdown("""
            <div class="metric-card" style="border-left: 6px solid #d32f2f; background:#ffebee;">
                <div style="font-weight:700; color:#c62828; font-size:1.15rem;">Candidate Elimination Version Space</div>
                <div style="font-size:1.3rem; font-weight:800; color:#b71c1c; margin:10px 0;">
                    ⚠️ Version Space is EMPTY (Collapsed)
                </div>
                <div style="font-size:0.85rem; color:#444; line-height:1.5;">
                    • <b>Status:</b> $S = \\emptyset$ or $G = \\emptyset$ (No consistent hypothesis exists).<br>
                    • <b>Cause:</b> Inconsistent or noisy training data was encountered.<br>
                    • <b>Limitation:</b> Candidate Elimination <u>assumes noise-free data</u>. A single mislabeled example collapses the entire Version Space!
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            s_str = ", ".join([f"⟨{', '.join(h)}⟩" for h in sorted(list(final_s))])
            g_str = "<br>".join([f"• <code>⟨{', '.join(h)}⟩</code>" for h in sorted(list(final_g))])
            st.markdown(f"""
            <div class="metric-card" style="border-left: 6px solid #1976d2;">
                <div style="font-weight:700; color:#0d47a1; font-size:1.15rem;">Candidate Elimination Boundaries</div>
                <div style="font-size:0.92rem; color:#333; margin:8px 0;">
                    <b>Specific Boundary (S):</b><br>
                    <span style="font-family:monospace; font-weight:700; color:#1565c0;">{s_str}</span>
                </div>
                <div style="font-size:0.92rem; color:#333; margin:8px 0;">
                    <b>General Boundary (G):</b><br>
                    {g_str}
                </div>
                <div style="font-size:0.85rem; color:#444; margin-top:6px;">
                    • <b>Version Space:</b> Contains all hypotheses $h$ satisfying $S \\le_g h \\le_g G$.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Step-by-Step Evolution Trace (Requirement 5)
    st.markdown("#### 🔍 Step-by-Step Trace of Hypothesis Updates")
    st.write("Track the exact updates to Find-S and Candidate Elimination after each training example:")

    trace_tabs = st.tabs(["📜 Candidate Elimination Step-by-Step", "📜 Find-S Step-by-Step"])

    with trace_tabs[0]:
        st.markdown("##### Candidate Elimination Version Space Updates (Step-by-Step)")
        ce_rows = []
        for step in ce.history:
            step_num = step["step"]
            if step_num == 0:
                inst_str = "Initialization"
                lbl_str = "N/A"
            else:
                ex_f = step["example"]
                inst_str = f"({ex_f['SoilType']}, {ex_f['Climate']}, {ex_f['Moisture']}, {ex_f['Rainfall']})"
                lbl_str = "✅ Yes (+)" if step["label"] in ["Yes", "+"] else "❌ No (-)"

            s_items = ", ".join([f"⟨{', '.join(h)}⟩" for h in sorted(list(step["S"]))]) or "∅ (Empty)"
            g_items = ", ".join([f"⟨{', '.join(h)}⟩" for h in sorted(list(step["G"]))]) or "∅ (Empty)"

            ce_rows.append({
                "Step": f"Step {step_num}",
                "Instance x": inst_str,
                "Label": lbl_str,
                "S Boundary Set": s_items,
                "G Boundary Set": g_items,
                "Action Taken": step["action"]
            })
        st.dataframe(pd.DataFrame(ce_rows), use_container_width=True, hide_index=True)

    with trace_tabs[1]:
        st.markdown("##### Find-S Hypothesis Updates (Step-by-Step)")
        finds_rows = []
        for step in find_s.history:
            step_num = step["step"]
            if step_num == 0:
                inst_str = "Initialization"
                lbl_str = "N/A"
            else:
                ex_f = step["example"]
                inst_str = f"({ex_f['SoilType']}, {ex_f['Climate']}, {ex_f['Moisture']}, {ex_f['Rainfall']})"
                lbl_str = "✅ Yes (+)" if step["label"] in ["Yes", "+"] else "❌ No (-)"

            h_str = f"⟨{', '.join(step['hypothesis'])}⟩"
            finds_rows.append({
                "Step": f"Step {step_num}",
                "Instance x": inst_str,
                "Label": lbl_str,
                "Current Hypothesis h": h_str,
                "Action Taken": step["action"]
            })
        st.dataframe(pd.DataFrame(finds_rows), use_container_width=True, hide_index=True)

    # 5. Interactive Concept Instance Query Classifier
    st.markdown("#### 🧪 Test Unseen Instances Against the Learned Concept")
    st.write("Select attribute values for an unseen farm profile and see how Find-S and Candidate Elimination classify it:")

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        q_soil = st.selectbox("Query SoilType", DEFAULT_ATTRIBUTES["SoilType"], key="q_soil")
    with t_col2:
        q_clim = st.selectbox("Query Climate", DEFAULT_ATTRIBUTES["Climate"], key="q_clim")
    with t_col3:
        q_moist = st.selectbox("Query Moisture", DEFAULT_ATTRIBUTES["Moisture"], key="q_moist")
    with t_col4:
        q_rain = st.selectbox("Query Rainfall", DEFAULT_ATTRIBUTES["Rainfall"], key="q_rain")

    query_instance = {"SoilType": q_soil, "Climate": q_clim, "Moisture": q_moist, "Rainfall": q_rain}

    if st.button("🔍 Classify Query Instance", use_container_width=True):
        find_s_pred = find_s.predict(query_instance)
        ce_pred = ce.predict(query_instance)

        qc1, qc2 = st.columns(2)
        with qc1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9rem; color:#555; font-weight:600;">Find-S Prediction</div>
                <div style="font-size:1.6rem; font-weight:800; color:{'#2e7d32' if find_s_pred == 'Yes' else '#c62828'};">
                    {find_s_pred}
                </div>
                <div style="font-size:0.82rem; color:#777;">
                    Matches specific hypothesis ⟨{', '.join(final_h)}⟩: {'True' if find_s_pred == 'Yes' else 'False'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with qc2:
            ce_color = "#2e7d32" if "Yes" in ce_pred else ("#c62828" if "No" in ce_pred else "#f57c00")
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9rem; color:#555; font-weight:600;">Candidate Elimination Prediction</div>
                <div style="font-size:1.4rem; font-weight:800; color:{ce_color};">
                    {ce_pred}
                </div>
                <div style="font-size:0.82rem; color:#777;">
                    Version Space agreement across all bounding hypotheses
                </div>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# VIEW 5: SEQUENTIAL COVERING (MODULE 2: RULE LEARNING)
# =====================================================================
elif nav_choice == "📜 Sequential Covering (Rule Learning)":
    st.markdown('<div class="section-title">📜 Sequential Covering (Module 2: Rule Learning)</div>', unsafe_allow_html=True)
    st.write("Induce transparent, human-interpretable IF–THEN production rules from categorical crop and climate patterns using greedy Learn-One-Rule beam search.")

    # Educational Disclaimer (Requirement 12)
    st.markdown("""
    <div class="clustering-disclaimer" style="border-left-color: #ff9800; background: #fff3e0; color: #e65100;">
        <b>🎓 Educational Machine Learning Demonstration Notice:</b><br>
        Sequential Covering is a classical symbolic rule-learning algorithm from machine learning theory (Tom Mitchell, <i>Machine Learning</i>, Ch. 10).<br>
        <b>These learned IF–THEN rules are educational pedagogical demonstrations on categorical data and do NOT establish verified agricultural suitability.</b><br>
        In production agronomy, crop productivity depends on complex micro-biological soil chemistry, diurnal temperature variations, and cultivar genetics.
    </div>
    """, unsafe_allow_html=True)

    # 1. Rule Learning Procedure Explanation (Requirement 5)
    st.markdown("#### ⚙️ The Sequential Covering Procedure")
    st.markdown("""
    Sequential Covering learns an ordered disjunction of conjunctive IF–THEN rules directly from training examples:
    1. **Target Selection**: Select a target crop class $C$ (or iterate sequentially through all classes).
    2. **Learn-One-Rule (Greedy Beam Search)**:
       - Start with an empty antecedent (`IF True THEN Crop = C`), which covers all examples.
       - Greedily specialize the rule by testing all possible `Attribute = Value` tests.
       - Select the condition that maximizes **Rule Precision** $p = \\frac{\\text{pos}}{\\text{pos} + \\text{neg}}$, breaking ties with higher positive coverage.
       - Stop specializing once the rule excludes all negative examples (100% pure) or reaches maximum conditions.
    3. **Add the Rule**: Append the induced rule to the learned rule set.
    4. **Remove Covered Examples**: Strip all positive examples covered by this rule from the remaining training pool (the "covering" step).
    5. **Iterate**: Repeat steps 2–4 until the stopping condition (all positives covered or minimum coverage reached) is met.
    """)

    # 2. Categorical Dataset Derivation & Documentation (Requirement 4)
    with st.expander("📘 Categorical Feature Derivation & Binning Thresholds (from Crop_recommendation.csv)", expanded=False):
        st.markdown("""
        To enable fast, beginner-friendly rule induction, continuous numeric features from `Crop_recommendation.csv` were discretized into categorical bins based on standard agronomic thresholds:
        - **MoistureLevel** (from relative humidity %):
          - `High`: humidity > 75% *(wetland / tropical monsoon air)*
          - `Medium`: 50% <= humidity <= 75% *(moderate ambient humidity)*
          - `Low`: humidity < 50% *(dry / arid atmosphere)*
        - **RainfallLevel** (from seasonal precipitation mm):
          - `High`: rainfall > 150 mm *(flood irrigation / heavy rainfall)*
          - `Moderate`: 70 mm <= rainfall <= 150 mm *(temperate rainfall)*
          - `Low`: rainfall < 70 mm *(semi-arid / drought conditions)*
        - **NutrientProfile** (from Nitrogen N and Potassium K in kg/ha):
          - `NitrogenRich`: N >= 80 kg/ha *(heavy vegetative foliage demand, e.g. Cotton)*
          - `PotassiumRich`: K >= 100 kg/ha *(orchard fruit sugar accumulation, e.g. Apple)*
          - `Balanced`: N < 80 and K < 100 *(standard nutrient ratio, e.g. Rice, Chickpea, Coffee)*
        - **TemperatureLevel** (from ambient temperature °C):
          - `Warm`: temperature > 24.0°C *(tropical / subtropical warmth)*
          - `Cool`: temperature <= 24.0°C *(temperate / montane climate)*
        """)

    # 3. Training Dataset Display
    st.markdown("#### 📋 Categorical Training Dataset (16 Documented Agricultural Instances)")
    df_cat = pd.DataFrame(SC_DATASET)
    st.dataframe(df_cat, use_container_width=True, hide_index=True)

    # 4. Interactive Rule Induction Controls (Requirement 3 & 6)
    st.markdown("#### 🔍 Induce IF–THEN Rules from Data")
    rule_mode = st.radio(
        "Choose Rule Induction Mode:",
        [
            "🎯 Target Specific Crop (One-vs-Rest Rule Learning)",
            "🌐 Multi-Class Decision List (All 5 Crops + Default Fallback)"
        ],
        index=0,
        horizontal=True
    )

    sc_engine = SequentialCovering(SC_ATTRIBUTES)

    if rule_mode == "🎯 Target Specific Crop (One-vs-Rest Rule Learning)":
        target_crop = st.selectbox("Select Target Crop Class:", ["Rice", "Apple", "Cotton", "Chickpea", "Coffee"], index=0)
        learned_rules = sc_engine.fit_target_class(SC_DATASET, target_class=target_crop)
        rule_context = f"Target Crop: '{target_crop}'"
    else:
        learned_rules = sc_engine.fit_all_classes(SC_DATASET, default_class="Rice")
        rule_context = "Multi-Class Decision List (Ordered Priority)"

    # 5. Display Learned Rules with Training Metrics (Requirement 6)
    st.markdown(f"##### 📜 Learned Rules for {rule_context}")
    st.caption("⚠️ **Training-Set Metrics Note**: Coverage and precision shown below represent empirical fit on the training examples, NOT independent test performance.")

    if not learned_rules:
        st.info("No rules were learned for the selected configuration.")
    else:
        for idx, rule in enumerate(learned_rules, 1):
            is_default = len(rule.conditions) == 0
            badge_color = "#6a1b9a" if is_default else "#1b5e20"
            border_color = "#ab47bc" if is_default else "#2e7d32"
            
            cond_display = rule.to_string()
            pos_count = rule.pos_covered_count
            neg_count = rule.neg_covered_count
            cov_ids = rule.covered_instance_ids
            prec_val = rule.precision * 100

            st.markdown(f"""
            <div class="metric-card" style="border-left: 6px solid {border_color}; margin-bottom: 0.8rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:700; color:{badge_color}; font-size:1.05rem;">
                        Rule #{idx}: <code style="font-size:1rem; color:#111;">{cond_display}</code>
                    </div>
                    <span style="background:#e8f5e9; color:#2e7d32; font-weight:700; padding:3px 10px; border-radius:12px; font-size:0.85rem;">
                        {'Default Fallback Rule' if is_default else f'Precision: {prec_val:.1f}%'}
                    </span>
                </div>
                <div style="font-size:0.88rem; color:#444; margin-top:8px;">
                    • <b>Positive Examples Covered:</b> {pos_count} instance(s) {f'(IDs: {cov_ids})' if cov_ids else ''}<br>
                    • <b>Negative Examples Covered:</b> {neg_count} instance(s) (Purity: {'100% Pure' if neg_count == 0 else f'{prec_val:.1f}%'})<br>
                    • <b>Evaluation Type:</b> <span style="color:#d84315; font-style:italic;">Training-Set Empirical Fit</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Overall Training-Set Evaluation
        eval_metrics = sc_engine.evaluate_training_set(SC_DATASET, learned_rules)
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.metric("Total Learned Rules", len(learned_rules))
        with col_e2:
            st.metric("Training Set Accuracy", f"{eval_metrics['accuracy']:.1f}%", help="Empirical fit across all 16 training instances")
        with col_e3:
            st.metric("Unmatched Instances", eval_metrics["unmatched"])

    # 6. Interactive Query Rule Matcher (Requirement 8 & 9)
    st.markdown("#### 🧪 Test Unseen Farm Profiles Against Learned Rules")
    st.write("Select categorical attributes for a farm and see which IF–THEN rule matches:")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        q_nutr = st.selectbox("NutrientProfile", SC_ATTRIBUTES["NutrientProfile"], key="sc_q_nutr")
    with m_col2:
        q_moist = st.selectbox("MoistureLevel", SC_ATTRIBUTES["MoistureLevel"], key="sc_q_moist")
    with m_col3:
        q_rain = st.selectbox("RainfallLevel", SC_ATTRIBUTES["RainfallLevel"], key="sc_q_rain")
    with m_col4:
        q_temp = st.selectbox("TemperatureLevel", SC_ATTRIBUTES["TemperatureLevel"], key="sc_q_temp")

    use_fallback = st.checkbox("Include Default Fallback Rule when Unmatched?", value=False, help="Uncheck to test how the system gracefully handles unmatched examples.")

    query_profile = {
        "NutrientProfile": q_nutr,
        "MoistureLevel": q_moist,
        "RainfallLevel": q_rain,
        "TemperatureLevel": q_temp
    }

    if st.button("🔍 Match Rules Against Query Farm Profile", use_container_width=True):
        active_rule_list = learned_rules if use_fallback else [r for r in learned_rules if len(r.conditions) > 0]
        pred_crop, matched_rule = sc_engine.predict(query_profile, rules=active_rule_list, fallback_class=None)

        if matched_rule is not None:
            is_def_matched = len(matched_rule.conditions) == 0
            crop_icon = CROP_ICONS.get(pred_crop.lower(), "🌱")
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1.1rem; color: #2e7d32; font-weight: 600;">MATCHED RULE PREDICTION</div>
                <div class="crop-name">{crop_icon} {pred_crop.upper()}</div>
                <div style="font-size:1.05rem; color:#1b5e20; font-family:monospace; margin-top:8px;">
                    Matched: <b>{matched_rule.to_string()}</b>
                </div>
                <div class="disclaimer-text">
                    Rule matched with {matched_rule.precision*100:.1f}% training precision. (Educational symbolic demonstration).
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Graceful Unmatched Example Handling (Requirement 9)
            st.markdown("""
            <div class="clustering-disclaimer" style="border-left-color: #d32f2f; background: #ffebee; color: #c62828;">
                <b>⚠️ No Learned Rule Matched This Profile:</b><br>
                The selected combination does not satisfy the antecedent constraints of any learned IF–THEN rule in the current rule set.<br>
                • <b>Cause:</b> The categorical feature subspace was not covered by the training examples.<br>
                • <b>Handling:</b> In production rule systems, unmatched instances are handled via a designated default fallback rule, majority class voting, or nearest-neighbor distance mapping.
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# VIEW 6: ALGORITHM EXPLANATIONS
# =====================================================================
elif nav_choice == "🧠 Algorithm Explanations":
    st.markdown('<div class="section-title">🧠 Machine Learning (ML-II) Algorithm Explanations</div>', unsafe_allow_html=True)
    st.write("Understand how each of the four algorithms works, their theoretical foundation, and why their performance varies on the crop recommendation problem.")

    algos = [
        {
            "name": "1. Decision Stump (Weak Learner)",
            "badge": "Single Rule Classifier • max_depth=1",
            "accuracy": f"{metrics_data['models']['Decision Stump']['accuracy']*100:.2f}%" if metrics_data else "9.09%",
            "desc": """
            - **What it is**: A Decision Stump is the simplest possible decision tree: it has exactly **one split** (depth = 1) along a single feature.
            - **How it works**: It scans all features (e.g., Rainfall, Nitrogen) and finds the single threshold that gives the highest information gain.
            - **Why it performs at ~9%**: With 22 distinct crop categories, a single binary question can only separate samples into at most two groups. Thus, it cannot solve multi-class non-linear agricultural boundaries on its own. In ensemble theory, it serves as the foundational **Weak Learner**.
            """
        },
        {
            "name": "2. Bagging Classifier (Bootstrap Aggregating)",
            "badge": "Parallel Ensemble • Variance Reduction",
            "accuracy": f"{metrics_data['models']['Bagging Classifier']['accuracy']*100:.2f}%" if metrics_data else "99.09%",
            "desc": """
            - **What it is**: An ensemble meta-estimator that trains multiple base estimators (here, 100 deep decision trees) in parallel.
            - **How it works**:
              1. **Bootstrapping**: Generates 100 random subsets of training data with replacement.
              2. **Independent Fitting**: Trains a separate decision tree on each subset.
              3. **Aggregation**: Gathers predictions from all 100 trees and determines the final crop via majority vote.
            - **Why it performs at ~99%**: High-depth decision trees have high variance and tend to overfit locally. Averaging 100 diverse trees smooths out errors and reduces variance dramatically without increasing bias.
            """
        },
        {
            "name": "3. Random Forest Classifier",
            "badge": "State-of-the-Art Ensemble • Feature Subspacing",
            "accuracy": f"{metrics_data['models']['Random Forest']['accuracy']*100:.2f}%" if metrics_data else "99.55%",
            "desc": """
            - **What it is**: An advanced evolution of Bagging designed by Leo Breiman and Adele Cutler.
            - **How it works**:
              - In addition to bootstrap sampling of rows, Random Forest injects **random feature subspace sampling** at every node split.
              - Instead of choosing the best split from all 7 features, each node considers only a random subset of $\\sqrt{7} \\approx 2$ to $3$ features.
            - **Why it achieves peak accuracy (~99.55%)**: This decorrelates the individual trees. If one feature (e.g., Rainfall) is universally dominant, normal bagging might still produce similar trees. Random Forest forces trees to discover subtle interactions across Nitrogen, Potassium, and Temperature, yielding the most robust classifier.
            """
        },
        {
            "name": "4. AdaBoost Classifier (Adaptive Boosting)",
            "badge": "Sequential Ensemble • Bias Reduction",
            "accuracy": f"{metrics_data['models']['AdaBoost Classifier']['accuracy']*100:.2f}%" if metrics_data else "31.36%",
            "desc": """
            - **What it is**: A sequential boosting algorithm invented by Freund and Schapire that iteratively converts weak learners into a stronger committee.
            - **How it works**:
              1. Starts with equal sample weights and trains a base decision stump.
              2. Samples that were **misclassified** receive higher weights; correctly classified samples receive lower weights.
              3. The next stump focuses heavily on the hard-to-classify samples.
              4. Final prediction is a weighted vote based on each stump's stage accuracy.
            - **Why it achieves ~31.36%**: While boosting elevates the stump from 9% to 31%, 100 stumps with depth=1 still struggle to approximate complex, non-linear 22-class hyperplanes compared to full-depth trees.
            """
        }
    ]

    for item in algos:
        st.markdown(f"""
        <div class="algo-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="color:#1b5e20; margin:0;">{item['name']}</h3>
                <span style="background:#e8f5e9; color:#2e7d32; font-weight:700; padding:4px 12px; border-radius:12px; border:1px solid #81c784;">
                    Test Accuracy: {item['accuracy']}
                </span>
            </div>
            <div style="color:#558b2f; font-weight:600; font-size:0.9rem; margin-top:4px;">{item['badge']}</div>
            <div style="margin-top:10px; color:#333; line-height:1.6;">
                {item['desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#688f6c; font-size:0.85rem; padding:10px;">
    🌾 <b>AgriSmart Crop Recommendation System</b> • Machine Learning in Precision Agriculture • Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
