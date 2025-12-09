import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load data
print("Loading health data...")
df = pd.read_csv('health_data_synthetic.csv')

print(f"Loaded {len(df)} days of data\n")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("First 5 days:")
print(df[['date', 'sleep_hours', 'mood', 'stress', 'productivity_hours']].head())
print()

# Create target variable
print("Creating target variable: TOMORROW'S MOOD")
print("-" * 60)

df['tomorrow_mood'] = df['mood'].shift(-1)
df = df[:-1].copy()

print("Created 'tomorrow_mood' column")
print(f"Training examples: {len(df)} (lost 1 day at the end)\n")

print("Example: Today's data -> Tomorrow's mood")
print(df[['date', 'sleep_hours', 'stress', 'mood', 'tomorrow_mood']].head(10))
print()

# Feature engineering
print("Engineering features...")

# Rolling averages
df['sleep_avg_3d'] = df['sleep_hours'].rolling(window=3, min_periods=1).mean()
df['sleep_avg_7d'] = df['sleep_hours'].rolling(window=7, min_periods=1).mean()
df['stress_avg_3d'] = df['stress'].rolling(window=3, min_periods=1).mean()
df['mood_avg_3d'] = df['mood'].rolling(window=3, min_periods=1).mean()
df['steps_avg_3d'] = df['steps'].rolling(window=3, min_periods=1).mean()

# Recent changes
df['sleep_change'] = df['sleep_hours'].diff().fillna(0)
df['stress_change'] = df['stress'].diff().fillna(0)
df['mood_change'] = df['mood'].diff().fillna(0)
df['steps_change'] = df['steps'].diff().fillna(0)

# Volatility
df['sleep_std_7d'] = df['sleep_hours'].rolling(window=7, min_periods=1).std().fillna(0)
df['stress_std_3d'] = df['stress'].rolling(window=3, min_periods=1).std().fillna(0)

# Day of week
df['day_of_week_num'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)

# Interaction features
df['sleep_stress_interaction'] = df['sleep_hours'] * df['stress']
df['productivity_per_step'] = df['productivity_hours'] / (df['steps'] + 1)

# Cumulative stress
df['cumulative_stress'] = df['stress'].rolling(window=7, min_periods=1).sum()

print("Created 17 engineered features")
print()

# Select features
print("Selecting features for the model...")

feature_columns = [
    'sleep_hours',
    'steps',
    'water_ml',
    'mood',
    'stress',
    'productivity_hours',
    'sleep_avg_3d',
    'sleep_avg_7d',
    'stress_avg_3d',
    'mood_avg_3d',
    'steps_avg_3d',
    'sleep_change',
    'stress_change',
    'mood_change',
    'steps_change',
    'sleep_std_7d',
    'stress_std_3d',
    'is_weekend',
    'sleep_stress_interaction',
    'productivity_per_step',
    'cumulative_stress'
]

X = df[feature_columns].copy()
y = df['tomorrow_mood'].copy()

print(f"Using {len(feature_columns)} features:")
for i, col in enumerate(feature_columns, 1):
    print(f"   {i}. {col}")
print()

missing = X.isnull().sum().sum()
if missing > 0:
    print(f"Found {missing} missing values - filling with 0...")
    X = X.fillna(0)
else:
    print("No missing values")
print()

# Split data
print("Splitting data into train (80%) and test (20%) sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False,
    random_state=42
)

print(f"Training set: {len(X_train)} days")
print(f"Test set: {len(X_test)} days")
print()

# Normalize data
print("Normalizing features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features normalized")
print()

# Train model
print("Training Random Forest Regressor...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Training in progress...")
model.fit(X_train_scaled, y_train)

print("Model trained successfully")
print()

# Evaluate model
print("EVALUATING MODEL PERFORMANCE")
print("=" * 60)

train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
train_r2 = r2_score(y_train, train_predictions)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

print("TRAINING SET PERFORMANCE:")
print(f"  MAE (Mean Absolute Error): {train_mae:.3f} mood points")
print(f"  RMSE (Root Mean Squared Error): {train_rmse:.3f}")
print(f"  R2 Score: {train_r2:.3f} ({train_r2*100:.1f}% variance explained)")
print()

print("TEST SET PERFORMANCE (unseen data):")
print(f"  MAE (Mean Absolute Error): {test_mae:.3f} mood points")
print(f"  RMSE (Root Mean Squared Error): {test_rmse:.3f}")
print(f"  R2 Score: {test_r2:.3f} ({test_r2*100:.1f}% variance explained)")
print()

print("INTERPRETATION:")
if test_mae < 0.5:
    print(f"  EXCELLENT - Predictions within +/-{test_mae:.2f} mood points on average")
elif test_mae < 0.8:
    print(f"  GOOD - Predictions within +/-{test_mae:.2f} mood points on average")
elif test_mae < 1.2:
    print(f"  FAIR - Predictions within +/-{test_mae:.2f} mood points on average")
else:
    print(f"  POOR - Predictions off by +/-{test_mae:.2f} mood points on average")

print()

# Feature importance analysis
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)
print("Which features matter most for predicting mood?\n")

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 60)
for i, row in feature_importance_df.head(10).iterrows():
    bar = '#' * int(row['importance'] * 200)
    print(f"{row['feature']:30s} {bar} {row['importance']:.4f}")
print()

# Visualize results
print("Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Mood Prediction Model - Performance Analysis', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_test, test_predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax1.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Mood', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted Mood', fontsize=11, fontweight='bold')
ax1.set_title(f'Actual vs Predicted Mood (Test Set)\nMAE: {test_mae:.3f}', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 5.5)
ax1.set_ylim(0.5, 5.5)

# Plot 2: Prediction Error Distribution
ax2 = axes[0, 1]
errors = y_test.values - test_predictions
ax2.hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title(f'Distribution of Prediction Errors\nMean Error: {errors.mean():.3f}', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Feature Importance
ax3 = axes[1, 0]
top_features = feature_importance_df.head(10)
y_pos = np.arange(len(top_features))
ax3.barh(y_pos, top_features['importance'], color='coral', edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_features['feature'])
ax3.invert_yaxis()
ax3.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax3.set_title('Top 10 Most Important Features', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Predictions Over Time
ax4 = axes[1, 1]
test_indices = y_test.index
ax4.plot(test_indices, y_test.values, 'o-', label='Actual Mood', linewidth=2, markersize=6)
ax4.plot(test_indices, test_predictions, 's-', label='Predicted Mood', linewidth=2, markersize=6, alpha=0.7)
ax4.fill_between(test_indices, y_test.values, test_predictions, alpha=0.2, color='gray')
ax4.set_xlabel('Day Index', fontsize=11, fontweight='bold')
ax4.set_ylabel('Mood (1-5)', fontsize=11, fontweight='bold')
ax4.set_title('Actual vs Predicted Mood Over Time (Test Set)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mood_prediction_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved: mood_prediction_results.png")
print()

# Save model
print("Saving model and artifacts...")

with open('mood_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved: mood_prediction_model.pkl")

with open('mood_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved: mood_scaler.pkl")

with open('mood_feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("Feature columns saved: mood_feature_columns.pkl")

feature_importance_df.to_csv('feature_importance.csv', index=False)
print("Feature importance saved: feature_importance.csv")
print()

# Test on new data
print("TESTING MODEL ON NEW DATA")
print("=" * 60)

today_data = {
    'sleep_hours': 6.5,
    'steps': 7500,
    'water_ml': 2000,
    'mood': 3,
    'stress': 3,
    'productivity_hours': 5.5,
    'sleep_avg_3d': 6.8,
    'sleep_avg_7d': 7.0,
    'stress_avg_3d': 2.8,
    'mood_avg_3d': 3.2,
    'steps_avg_3d': 7800,
    'sleep_change': -0.5,
    'stress_change': 0.5,
    'mood_change': -0.5,
    'steps_change': -300,
    'sleep_std_7d': 0.8,
    'stress_std_3d': 0.7,
    'is_weekend': 0,
    'sleep_stress_interaction': 6.5 * 3,
    'productivity_per_step': 5.5 / 7500,
    'cumulative_stress': 18
}

print("Today's data:")
print(f"   Sleep: {today_data['sleep_hours']}h")
print(f"   Steps: {today_data['steps']}")
print(f"   Mood: {today_data['mood']}/5")
print(f"   Stress: {today_data['stress']}/5")
print(f"   Productivity: {today_data['productivity_hours']}h")
print()

today_df = pd.DataFrame([today_data])[feature_columns]
today_scaled = scaler.transform(today_df)
predicted_mood = model.predict(today_scaled)[0]

print("PREDICTION FOR TOMORROW:")
print(f"   Expected Mood: {predicted_mood:.1f}/5.0")
print()

if predicted_mood >= 4.0:
    print("   Outlook: POSITIVE - Tomorrow looks good")
    print("   Tip: Keep up your current healthy habits")
elif predicted_mood >= 3.0:
    print("   Outlook: NEUTRAL - Average day expected")
    print("   Tip: Consider getting extra sleep tonight")
elif predicted_mood >= 2.0:
    print("   Outlook: CONCERNING - Tomorrow may be challenging")
    print("   Recommendations:")
    print("      - Prioritize 8+ hours sleep tonight")
    print("      - Plan lighter workload tomorrow")
    print("      - Practice stress management techniques")
else:
    print("   Outlook: POOR - High risk for difficult day")
    print("   Urgent Recommendations:")
    print("      - Get full rest tonight (8-9 hours)")
    print("      - Reduce commitments for tomorrow")
    print("      - Consider reaching out for support")

print()

# Test multiple scenarios
print("SCENARIO TESTING")
print("=" * 60)

scenarios = [
    {
        'name': 'Great Day',
        'data': {'sleep_hours': 8.5, 'mood': 5, 'stress': 1, 'productivity_hours': 7,
                'sleep_avg_3d': 8.2, 'stress_avg_3d': 1.2}
    },
    {
        'name': 'Exhausted',
        'data': {'sleep_hours': 5.0, 'mood': 2, 'stress': 4, 'productivity_hours': 3,
                'sleep_avg_3d': 5.5, 'stress_avg_3d': 4.2}
    },
    {
        'name': 'Recovering from Stress',
        'data': {'sleep_hours': 8.0, 'mood': 3, 'stress': 3, 'productivity_hours': 4,
                'sleep_avg_3d': 6.5, 'stress_avg_3d': 4.5}
    }
]

for scenario in scenarios:
    test_data = today_data.copy()
    test_data.update(scenario['data'])
    test_data['sleep_stress_interaction'] = test_data['sleep_hours'] * test_data['stress']
    
    test_df = pd.DataFrame([test_data])[feature_columns]
    test_scaled = scaler.transform(test_df)
    pred = model.predict(test_scaled)[0]
    
    print(f"\n{scenario['name']}:")
    print(f"   Today: Sleep={scenario['data']['sleep_hours']}h, Stress={scenario['data']['stress']}/5")
    print(f"   -> Predicted Tomorrow's Mood: {pred:.1f}/5")

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print()
print("Files created:")
print("   1. mood_prediction_model.pkl    - Trained model")
print("   2. mood_scaler.pkl              - Data normalizer")
print("   3. mood_feature_columns.pkl     - Feature order")
print("   4. feature_importance.csv       - Feature rankings")
print("   5. mood_prediction_results.png  - Visualizations")
print()
print("Next: Build a Flask API or integrate into your app.")