import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── Paths ─────────────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('data/phone_usage.csv')
print("📊 Dataset shape:", df.shape)
print(df['risk_level'].value_counts(), "\n")

X = df.drop('risk_level', axis=1)
y = df['risk_level']

# ── Scale Features (Critical for KNN) ────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── Elbow Method: Find Best K ─────────────────────────────────────────────────
error_rates = []
k_range = range(1, 26)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(np.mean(preds != y_test))

best_k = error_rates.index(min(error_rates)) + 1
print(f"✅ Best K = {best_k}  (Error: {min(error_rates):.4f})")

# Plot elbow
plt.figure(figsize=(10, 5))
plt.plot(k_range, error_rates, marker='o', color='steelblue', linewidth=2, markersize=7)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
plt.title('Elbow Method — Finding Optimal K', fontsize=15, fontweight='bold')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/elbow_curve.png', dpi=150)
plt.close()
print("📈 Elbow curve saved → plots/elbow_curve.png")

# ── Train Final Model ─────────────────────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(knn, X_scaled, y, cv=5)
print(f"🔁 Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Confusion Matrix Plot ─────────────────────────────────────────────────────
labels = ['Low', 'Medium', 'High']
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.close()
print("📈 Confusion matrix saved → plots/confusion_matrix.png")

# ── Feature Importance (via permutation proxy) ────────────────────────────────
feature_names = X.columns.tolist()
baseline_acc = np.mean(y_pred == y_test)
importances = []

for i, feat in enumerate(feature_names):
    X_perm = X_test.copy()
    X_perm[:, i] = np.random.permutation(X_perm[:, i])
    perm_acc = np.mean(knn.predict(X_perm) == y_test)
    importances.append(baseline_acc - perm_acc)

plt.figure(figsize=(9, 5))
colors = ['#e74c3c' if v > 0 else '#95a5a6' for v in importances]
bars = plt.barh(feature_names, importances, color=colors)
plt.xlabel('Accuracy Drop (higher = more important)')
plt.title('Feature Importance (Permutation Method)', fontsize=14, fontweight='bold')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.close()
print("📈 Feature importance saved → plots/feature_importance.png")

# ── EDA Plots ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('data/phone_usage.csv')

# Risk distribution
plt.figure(figsize=(7, 4))
order = ['Low', 'Medium', 'High']
palette = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
sns.countplot(x='risk_level', data=df_raw, order=order, palette=palette)
plt.title('Risk Level Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Risk Level'); plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/risk_distribution.png', dpi=150)
plt.close()

# Correlation heatmap
plt.figure(figsize=(9, 7))
corr = df_raw.drop('risk_level', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150)
plt.close()

# Screen time vs Sleep
plt.figure(figsize=(8, 5))
palette2 = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
sns.scatterplot(x='screen_time_hrs', y='sleep_hours',
                hue='risk_level', data=df_raw,
                palette=palette2, alpha=0.7, s=60,
                hue_order=['Low', 'Medium', 'High'])
plt.title('Screen Time vs Sleep Hours by Risk', fontsize=14, fontweight='bold')
plt.xlabel('Screen Time (hrs)'); plt.ylabel('Sleep Hours')
plt.tight_layout()
plt.savefig('plots/screen_vs_sleep.png', dpi=150)
plt.close()
print("📈 EDA plots saved → plots/")

# ── Save Model ────────────────────────────────────────────────────────────────
pickle.dump(knn, open('model/knn_model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
print("\n✅ Model saved → model/knn_model.pkl")
print("✅ Scaler saved → model/scaler.pkl")
print("\n🎉 Training complete! Run: streamlit run app.py")
