import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

# Load data
print("Loading health data...")
df = pd.read_csv('health_data_synthetic.csv')

print(f"Loaded {len(df)} days of data")
print(f"Columns: {list(df.columns)}\n")

print("First 5 days:")
print(df.head())
print()

# Prepare features
print("Preparing features for ML model...")

feature_columns = [
    'sleep_hours',
    'steps',
    'water_ml',
    'mood',
    'stress',
    'productivity_hours'
]

X = df[feature_columns].copy()

print(f"Using {len(feature_columns)} features:")
for col in feature_columns:
    print(f"   - {col}")
print()

# Check for missing data
missing = X.isnull().sum().sum()
if missing > 0:
    print(f"Found {missing} missing values - filling with median...")
    X = X.fillna(X.median())
else:
    print("No missing data")
print()

# Feature engineering
print("Engineering additional features...")

df['sleep_3day_avg'] = df['sleep_hours'].rolling(window=3, min_periods=1).mean()
df['stress_3day_avg'] = df['stress'].rolling(window=3, min_periods=1).mean()
df['mood_3day_avg'] = df['mood'].rolling(window=3, min_periods=1).mean()

df['sleep_change'] = df['sleep_hours'].diff().fillna(0)
df['stress_change'] = df['stress'].diff().fillna(0)
df['mood_change'] = df['mood'].diff().fillna(0)

engineered_features = [
    'sleep_3day_avg',
    'stress_3day_avg',
    'mood_3day_avg',
    'sleep_change',
    'stress_change',
    'mood_change'
]

for feat in engineered_features:
    X[feat] = df[feat]

print(f"Added {len(engineered_features)} engineered features")
print(f"Total features: {len(X.columns)}")
print()

# Normalize data
print("Normalizing data...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data normalized")
print(f"Shape: {X_scaled.shape}")
print()

# Train model
print("Training Isolation Forest model...")

model = IsolationForest(
    contamination=0.10,
    random_state=42,
    n_estimators=100,
    max_samples='auto',
    bootstrap=False,
    n_jobs=-1
)

model.fit(X_scaled)

print("Model trained successfully")
print()

# Make predictions
print("Making predictions on all 90 days...")

anomaly_labels = model.predict(X_scaled)
anomaly_scores = model.decision_function(X_scaled)

# Convert to 0-100 risk scale
min_score = anomaly_scores.min()
max_score = anomaly_scores.max()
risk_scores = 100 * (1 - (anomaly_scores - min_score) / (max_score - min_score))

df['anomaly'] = anomaly_labels
df['anomaly_score'] = anomaly_scores
df['risk_score'] = risk_scores.round(1)

print("Predictions complete")
print()

# Analyze results
print("RESULTS ANALYSIS")
print("=" * 60)

num_anomalies = (anomaly_labels == -1).sum()
num_normal = (anomaly_labels == 1).sum()

print(f"Normal days: {num_normal} ({num_normal/len(df)*100:.1f}%)")
print(f"Anomaly days: {num_anomalies} ({num_anomalies/len(df)*100:.1f}%)")
print()

print("DETECTED ANOMALIES:")
print("-" * 60)
anomaly_days = df[df['anomaly'] == -1][['date', 'day_of_week', 'sleep_hours',
                                         'mood', 'stress', 'risk_score', 'symptoms']]
print(anomaly_days.to_string(index=False))
print()

print("TOP 10 HIGHEST RISK DAYS:")
print("-" * 60)
top_risk = df.nlargest(10, 'risk_score')[['date', 'day_of_week', 'sleep_hours',
                                            'mood', 'stress', 'risk_score', 'symptoms']]
print(top_risk.to_string(index=False))
print()

# Visualize results
print("Creating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')

df['date_dt'] = pd.to_datetime(df['date'])

# Plot 1: Risk Score Over Time
ax1 = axes[0]
ax1.plot(df['date_dt'], df['risk_score'], linewidth=2, color='steelblue', label='Risk Score')
ax1.axhline(y=50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium Risk')
ax1.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Risk')
ax1.fill_between(df['date_dt'], df['risk_score'], alpha=0.3, color='steelblue')
ax1.scatter(df[df['anomaly']==-1]['date_dt'],
           df[df['anomaly']==-1]['risk_score'],
           color='red', s=100, zorder=5, label='Anomaly', marker='X')
ax1.set_ylabel('Risk Score (0-100)', fontsize=11, fontweight='bold')
ax1.set_title('Daily Health Risk Score', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Sleep Hours with Anomalies
ax2 = axes[1]
ax2.plot(df['date_dt'], df['sleep_hours'], linewidth=2, color='purple', label='Sleep Hours')
ax2.scatter(df[df['anomaly']==-1]['date_dt'],
           df[df['anomaly']==-1]['sleep_hours'],
           color='red', s=100, zorder=5, label='Anomaly', marker='X')
ax2.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Healthy (7h)')
ax2.set_ylabel('Sleep Hours', fontsize=11, fontweight='bold')
ax2.set_title('Sleep Pattern with Anomalies Highlighted', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Plot 3: Mood vs Stress
ax3 = axes[2]
scatter = ax3.scatter(df['stress'], df['mood'],
                     c=df['risk_score'], cmap='RdYlGn_r',
                     s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.scatter(df[df['anomaly']==-1]['stress'],
           df[df['anomaly']==-1]['mood'],
           color='red', s=200, marker='X',
           edgecolors='darkred', linewidth=2, label='Anomaly', zorder=5)
ax3.set_xlabel('Stress Level (1-5)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Mood (1-5)', fontsize=11, fontweight='bold')
ax3.set_title('Mood vs Stress (colored by Risk Score)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Risk Score', fontsize=10, fontweight='bold')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved: anomaly_detection_results.png")
print()

# Save model and artifacts
print("Saving model and scaler...")

with open('anomaly_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved: anomaly_model.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved: scaler.pkl")

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("Feature columns saved: feature_columns.pkl")
print()

df.to_csv('health_data_with_predictions.csv', index=False)
print("Results saved: health_data_with_predictions.csv")
print()

# Test on new data
print("TESTING MODEL ON NEW DATA")
print("=" * 60)

new_day = {
    'sleep_hours': 5.0,
    'steps': 3000,
    'water_ml': 1000,
    'mood': 2,
    'stress': 5,
    'productivity_hours': 3.0
}

print("New day data:")
for key, val in new_day.items():
    print(f"   {key}: {val}")
print()

new_day_full = new_day.copy()
new_day_full['sleep_3day_avg'] = new_day['sleep_hours']
new_day_full['stress_3day_avg'] = new_day['stress']
new_day_full['mood_3day_avg'] = new_day['mood']
new_day_full['sleep_change'] = -2.0
new_day_full['stress_change'] = 2.0
new_day_full['mood_change'] = -2.0

new_df = pd.DataFrame([new_day_full])[X.columns]
new_scaled = scaler.transform(new_df)

prediction = model.predict(new_scaled)[0]
score = model.decision_function(new_scaled)[0]
risk = 100 * (1 - (score - min_score) / (max_score - min_score))

print("PREDICTION:")
if prediction == -1:
    print("   ANOMALY DETECTED")
else:
    print("   Normal day")

print(f"   Risk Score: {risk:.1f}/100")
print()

if risk > 75:
    print("   HIGH RISK - Immediate attention needed")
    print("   Recommendations:")
    print("      - Prioritize sleep tonight (aim for 8+ hours)")
    print("      - Take breaks to reduce stress")
    print("      - Consider talking to someone")
elif risk > 50:
    print("   MEDIUM RISK - Monitor closely")
    print("   Recommendations:")
    print("      - Ensure adequate rest")
    print("      - Practice stress management")
else:
    print("   LOW RISK - Keep up good habits")

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print()
print("Files created:")
print("   1. anomaly_model.pkl          - Trained model")
print("   2. scaler.pkl                 - Data normalizer")
print("   3. feature_columns.pkl        - Feature order")
print("   4. anomaly_detection_results.png - Visualizations")
print("   5. health_data_with_predictions.csv - Results")
print()
print("Next step: Use these files in your app to predict new days.")