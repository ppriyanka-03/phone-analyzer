# 📱 Phone Dependency Risk Analyzer — KNN

A machine learning project that predicts smartphone addiction risk level
(**Low / Medium / High**) based on daily phone usage habits using the
**K-Nearest Neighbors (KNN)** algorithm.

---

## 🚀 Quick Start (3 Steps)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate Data & Train Model
```bash
python generate_data.py   # creates data/phone_usage.csv
python train.py           # trains model, saves plots
```

### Step 3 — Launch the App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
phone-dependency-analyzer/
├── generate_data.py     ← Synthetic dataset generator
├── train.py             ← KNN model training + evaluation
├── app.py               ← Streamlit web application
├── requirements.txt     ← Python dependencies
├── data/
│   └── phone_usage.csv  ← Generated after running generate_data.py
├── model/
│   ├── knn_model.pkl    ← Trained KNN model
│   └── scaler.pkl       ← Fitted StandardScaler
└── plots/               ← EDA & evaluation charts
    ├── risk_distribution.png
    ├── correlation_heatmap.png
    ├── screen_vs_sleep.png
    ├── elbow_curve.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 🔢 Features Used

| Feature              | Description                          |
|----------------------|--------------------------------------|
| `screen_time_hrs`    | Total daily screen time in hours     |
| `unlocks_per_day`    | Number of times phone is unlocked    |
| `night_usage`        | Uses phone after 11 PM (0/1)         |
| `social_media_hrs`   | Time spent on social media (hrs)     |
| `app_switches_hr`    | App switches per hour                |
| `notifications_day`  | Number of notifications per day      |
| `sleep_hours`        | Hours of sleep per night             |
| `age`                | User age                             |

**Target:** `risk_level` → Low / Medium / High

---

## 🧠 Algorithm: K-Nearest Neighbors

- **Distance Metric**: Euclidean
- **Scaling**: StandardScaler (required for KNN)
- **Optimal K**: Found via Elbow Method (test K = 1 to 25)
- **Evaluation**: Accuracy, F1-Score, Confusion Matrix, 5-Fold CV

---

## 📊 App Features

- 🎛️ Interactive sliders for all 8 input features
- 🟢🟡🔴 Risk classification with confidence percentages
- 💡 Personalized recommendations per risk level
- 📊 EDA plots (correlation heatmap, scatter plots, distributions)
- 📈 Model evaluation charts (elbow curve, confusion matrix)
- ℹ️ KNN explanation tab

---

## 🛠️ Tech Stack

- **Python** — Core language
- **scikit-learn** — KNN model, preprocessing, evaluation
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualizations
- **Streamlit** — Web application UI

---

## 📌 Learning Outcomes

- End-to-end ML pipeline (data → model → deployment)
- KNN classification with hyperparameter tuning
- Feature scaling importance
- Model evaluation metrics
- Building ML-powered web apps with Streamlit
=======
# phone-analyzer
>>>>>>> 8274cc07ad600749b8d0fed8a09612f332e2d248
