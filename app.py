import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phone Dependency Risk Analyzer",
    page_icon="📱",
    layout="wide"
)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    knn    = pickle.load(open('model/knn_model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl',    'rb'))
    return knn, scaler

try:
    knn, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .risk-card {
        padding: 2rem; border-radius: 16px;
        text-align: center; font-size: 1.8rem; font-weight: 700;
        margin-top: 1rem;
    }
    .low    { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .medium { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .high   { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .metric-box {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center; border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📱 Phone Dependency Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown("**Powered by K-Nearest Neighbors (KNN) Machine Learning**")
st.markdown("---")

if not model_loaded:
    st.error("⚠️ Model not found! Please run the setup first:")
    st.code("python generate_data.py\npython train.py", language="bash")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Risk Analyzer", "📊 EDA & Insights", "ℹ️ About KNN"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — RISK ANALYZER
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Your Phone Usage Habits")
    st.markdown("Move the sliders to reflect your average daily usage.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📲 Usage Patterns")
        screen_time   = st.slider("Daily Screen Time (hrs)",    0.0, 15.0, 5.0, 0.5)
        unlocks       = st.slider("Phone Unlocks per Day",      0,   250,  80)
        social_media  = st.slider("Social Media Time (hrs)",    0.0,  8.0, 2.0, 0.5)
        app_switches  = st.slider("App Switches per Hour",      0,   100,  25)

    with col2:
        st.markdown("#### 🌙 Lifestyle Factors")
        night_usage   = st.selectbox("Use Phone After 11 PM?", ["No", "Yes"])
        notifications = st.slider("Notifications per Day",     0,   400, 100)
        sleep_hours   = st.slider("Sleep Hours per Night",     2.0,  10.0, 7.0, 0.5)
        age           = st.number_input("Your Age", min_value=10, max_value=80, value=22)

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze My Risk", use_container_width=True, type="primary")

    if analyze_btn:
        night = 1 if night_usage == "Yes" else 0
        features = np.array([[screen_time, unlocks, night,
                               social_media, app_switches,
                               notifications, sleep_hours, age]])
        features_scaled = scaler.transform(features)

        prediction  = knn.predict(features_scaled)[0]
        proba       = knn.predict_proba(features_scaled)[0]
        classes     = knn.classes_

        # Risk card
        css_class = prediction.lower()
        icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[prediction]
        st.markdown(
            f'<div class="risk-card {css_class}">{icon} {prediction} Risk</div>',
            unsafe_allow_html=True
        )

        # Probabilities
        st.markdown("#### Confidence Scores")
        prob_cols = st.columns(len(classes))
        color_map = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        for i, (cls, prob) in enumerate(zip(classes, proba)):
            with prob_cols[i]:
                st.metric(label=f"{cls} Risk", value=f"{prob*100:.1f}%")

        # Advice
        st.markdown("#### 💡 Recommendations")
        if prediction == "High":
            st.error("""
**Your phone usage is at a concerning level.** Consider:
- 📵 Enable Screen Time / Digital Wellbeing limits
- 🔕 Turn off non-essential notifications
- 📵 Use 'Do Not Disturb' mode after 10 PM
- 🚶 Take regular phone-free breaks during the day
- 😴 Charge your phone outside the bedroom
            """)
        elif prediction == "Medium":
            st.warning("""
**You're borderline — small changes can make a big difference:**
- ⏰ Set a daily screen time goal (try reducing by 30 mins)
- 📱 Avoid phone during meals and first 30 mins of morning
- 🌙 Limit social media to designated time slots
            """)
        else:
            st.success("""
**Great job! Your phone habits look healthy. Keep it up:**
- ✅ Maintain your current sleep schedule
- ✅ Continue balancing digital and offline activities
- 📊 Re-check monthly to track any changes
            """)

        # Input summary
        with st.expander("📋 View Your Input Summary"):
            summary = {
                "Screen Time (hrs)": screen_time,
                "Unlocks/Day": unlocks,
                "Night Usage": night_usage,
                "Social Media (hrs)": social_media,
                "App Switches/hr": app_switches,
                "Notifications/Day": notifications,
                "Sleep Hours": sleep_hours,
                "Age": age
            }
            st.dataframe(pd.DataFrame(summary, index=["Your Values"]).T,
                         use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Exploratory Data Analysis")

    plots = {
        "Risk Level Distribution":    "plots/risk_distribution.png",
        "Feature Correlation Heatmap": "plots/correlation_heatmap.png",
        "Screen Time vs Sleep":        "plots/screen_vs_sleep.png",
        "Elbow Curve (Best K)":        "plots/elbow_curve.png",
        "Confusion Matrix":            "plots/confusion_matrix.png",
        "Feature Importance":          "plots/feature_importance.png",
    }

    all_exist = all(os.path.exists(p) for p in plots.values())
    if not all_exist:
        st.warning("Plots not found. Run `python train.py` to generate them.")
    else:
        cols = st.columns(2)
        for i, (title, path) in enumerate(plots.items()):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)

    # Dataset preview
    if os.path.exists('data/phone_usage.csv'):
        st.markdown("---")
        st.subheader("📄 Dataset Preview")
        df = pd.read_csv('data/phone_usage.csv')
        st.dataframe(df.head(20), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Features", df.shape[1] - 1)
        c3.metric("Classes", df['risk_level'].nunique())

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT KNN
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ How KNN Works in This Project")

    st.markdown("""
### 🔵 What is KNN?
**K-Nearest Neighbors** is a simple yet powerful classification algorithm.
Instead of learning explicit rules, it memorizes all training examples and
classifies new inputs by finding the **K most similar training points**
and taking a **majority vote**.

---
### ⚙️ Steps in This Project

| Step | Description |
|------|-------------|
| **1. Data Collection** | 600 synthetic records with 8 phone usage features |
| **2. Feature Scaling** | `StandardScaler` ensures all features are on equal footing |
| **3. Train/Test Split** | 80% training, 20% testing with stratification |
| **4. Elbow Method** | Test K from 1–25, pick K with lowest error rate |
| **5. Prediction** | For new input, find K nearest neighbors → majority vote |
| **6. Evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix |

---
### 📐 Distance Formula (Euclidean)
KNN measures similarity using:

```
d(A, B) = √[ (a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)² ]
```

The closer the distance, the more similar the users.

---
### ✅ Why KNN for Phone Dependency?
- **Intuitive**: "People with similar habits tend to have similar risks"
- **No assumptions**: Doesn't assume linear separability
- **Fast prototyping**: Great for small-to-medium datasets
- **Interpretable**: Easy to explain to non-technical users

---
### ⚠️ KNN Limitations
- Slow at prediction time for large datasets (must compute all distances)
- Sensitive to irrelevant features → use feature selection
- Requires feature scaling → we use `StandardScaler`
- Choosing wrong K → we use the Elbow Method
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Python · scikit-learn · Streamlit</center>",
    unsafe_allow_html=True
)
