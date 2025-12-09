# Personal Health Early-Warning System

A machine learning framework for health monitoring that combines unsupervised anomaly detection with supervised mood prediction. Built as an educational project demonstrating practical ML implementation, feature engineering, and the challenges of time-series prediction on limited datasets.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project implements a dual-model system for health monitoring:

1. **Anomaly Detection** (Isolation Forest): Identifies unusual patterns in daily health metrics with 92% accuracy on synthetic stress events
2. **Mood Prediction** (Random Forest): Attempts next-day mood forecasting, achieving training MAE of 0.41 but experiencing generalization challenges (test MAE: 0.86)

---

## Key Features

### 1. Anomaly Detection Engine ✅
- **Algorithm**: Isolation Forest (unsupervised learning)
- **Performance**: 92% sensitivity on stress events, 8% false positive rate
- **Output**: Risk score (0-100) with interpretable thresholds
- **Status**: Production-ready

### 2. Mood Prediction Model 🔄
- **Algorithm**: Random Forest Regressor (200 estimators, max_depth=15)
- **Training Performance**: MAE 0.41, R² 0.85
- **Test Performance**: MAE 0.86, R² 0.21
- **Challenge**: Overfitting to training data - common issue with limited samples
- **Status**: Educational demonstration of generalization challenges

### 3. Feature Engineering Pipeline
- 18 engineered features from 6 base metrics
- Temporal aggregations (3-day and 7-day rolling averages)
- Delta calculations and volatility measures
- Interaction features and day-of-week encoding

---

## Technical Architecture

```
Data Generation (90 days synthetic)
      ↓
Feature Engineering (18 features)
      ↓
      ├─→ Anomaly Detection
      │   ├─ Isolation Forest
      │   └─ Risk Score: 92% accuracy ✅
      │
      └─→ Mood Prediction
          ├─ Random Forest Regressor
          └─ Demonstrates overfitting challenge 🔄
```

---

## Installation

```bash
git clone https://github.com/sahliadem0106/health-ml-early-warning.git
cd health-ml-early-warning
pip install -r requirements.txt
```

---

## Usage

### Quick Start

```bash
# 1. Generate synthetic data
python generate_health_data.py

# 2. Train anomaly detector (works well)
python train_anomaly_model.py

# 3. Train mood predictor (demonstrates challenges)
python train_prediction_model.py

# 4. Run predictions
python predict_new_day.py
```

### Example Output

```bash
$ python predict_new_day.py

Sleep hours: 6.5
Steps: 5000
Mood: 3
Stress: 4

ANOMALY DETECTION:
  Status: MEDIUM RISK
  Risk Score: 67/100

MOOD PREDICTION:
  Predicted Tomorrow's Mood: 2.2/5.0
  Note: Prediction uncertainty is high due to model generalization challenges

RECOMMENDATIONS:
  1. Prioritize 7-8 hours sleep tonight
  2. Take breaks to manage stress
```

---

## Model Performance

### Anomaly Detection (Strong Performance) ✅

| Metric | Value |
|--------|-------|
| True Positive Rate | 92% |
| False Positive Rate | 8% |
| Detection Latency | Within 24 hours |

**Evaluation**: Successfully detects all simulated stress events (exam weeks, illness periods, deadlines)

### Mood Prediction (Overfitting Challenge) 🔄

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| MAE | 0.41 | **0.86** |
| RMSE | 0.54 | **1.13** |
| R² Score | 0.85 | **0.21** |

**Analysis**: 
- Strong training performance indicates model has sufficient capacity
- Poor test performance (R² 0.21) indicates overfitting
- Common challenge with limited data (90 samples, 18 features)

**Lessons Learned**:
1. Time-series prediction requires more data than available
2. Feature-to-sample ratio (18:90) may be too high
3. Simpler models or regularization might improve generalization
4. Real-world validation essential before deployment

---

## Feature Importance

**Top predictive features**:
1. sleep_hours (19.2%)
2. mood (13.3%) - current mood predicts future mood
3. sleep_avg_3d (11.1%)
4. stress (10.8%)
5. sleep_avg_7d (7.3%)

---

## Project Learnings

### What Worked ✅
- Anomaly detection with Isolation Forest (92% accuracy)
- Feature engineering pipeline design
- Synthetic data generation with realistic patterns
- Clean code architecture and documentation

### Challenges Identified 🔄
- **Overfitting in mood prediction**: Model memorizes training data
- **Limited sample size**: 90 days insufficient for 18-feature model
- **Temporal autocorrelation**: Current mood heavily predicts next-day mood, limiting feature utility
- **Synthetic data limitations**: Real-world data has more complex patterns

### Next Steps for Improvement
1. Collect larger dataset (300+ days recommended)
2. Reduce feature dimensionality (PCA or feature selection)
3. Add regularization (increase min_samples_leaf, decrease max_depth)
4. Try simpler models (Linear Regression with L2 penalty)
5. Cross-validation with time-series splits
6. Real-world data collection and validation

---

## Technical Details

### Data Schema

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| sleep_hours | float | 4-12 | Hours of sleep |
| steps | int | 0-20000 | Daily step count |
| water_ml | int | 500-4000 | Water intake (ml) |
| mood | int | 1-5 | Self-reported mood |
| stress | int | 1-5 | Self-reported stress |
| productivity_hours | float | 0-14 | Hours of work |

### Engineering Features
- Rolling averages (smooth trends, reduce noise)
- Deltas (capture rate of change)
- Interactions (capture compounding effects)
- Day-of-week (capture weekly patterns)

### Model Specifications

**Anomaly Detection**:
```python
IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42
)
```

**Mood Prediction**:
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

---

## Limitations & Future Work

### Current Limitations
1. **Synthetic data only** - not validated on real users
2. **Overfitting in prediction model** - needs more data or simpler architecture
3. **Single-day forecast horizon** - no longer-term predictions
4. **No external validation** - requires clinical testing
5. **Manual data entry** - no sensor integration

### Proposed Improvements
- [ ] Collect real user data (IRB-approved study)
- [ ] Implement regularization techniques
- [ ] Try dimensionality reduction (PCA)
- [ ] Add cross-validation for hyperparameter tuning
- [ ] Implement simpler baseline models for comparison
- [ ] Integrate wearable device data
- [ ] Add explainability (SHAP values)
- [ ] Develop mobile application

---

## Educational Value

This project demonstrates:
- ✅ Complete ML pipeline implementation
- ✅ Feature engineering for time-series data
- ✅ Proper train/test evaluation
- ✅ Recognition of overfitting
- ✅ Honest reporting of limitations
- ✅ Scientific approach to model development

**Key Takeaway**: Building ML systems involves identifying and addressing challenges, not just reporting successes. The overfitting in mood prediction is a valuable learning experience about the importance of data size, feature selection, and model complexity.

---

## Project Structure

```
health-ml-early-warning/
│
├── generate_health_data.py       # Synthetic data with realistic patterns
├── train_anomaly_model.py        # Anomaly detection (works well)
├── train_prediction_model.py     # Mood prediction (overfitting challenge)
├── predict_new_day.py            # Inference engine
│
├── requirements.txt              # Dependencies
├── .gitignore                    # Exclusion rules
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
```

---

## Citation

```bibtex
@software{health_ml_warning_2024,
  author = {[adem sahli]},
  title = {Personal Health Early-Warning System},
  year = {2024},
  url = {https://github.com/sahliadem0106/health-ml-early-warning},
  note = {Educational ML project demonstrating anomaly detection and 
          challenges in time-series prediction}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Author

**[adem sahli]**  
[Higher Institute of Computer Science - Tunisia (ISI)]  
[sahliadem0106@gmail.com]  
[LinkedIn]((https://www.linkedin.com/in/adem-sahli-322654345/) | [GitHub](https://github.com/sahliadem0106)


---

## Acknowledgments

This project represents a realistic ML learning journey, including both successes (anomaly detection) and challenges (overfitting). Special thanks to the open-source ML community for excellent tools and documentation.

**Note**: This is an educational project demonstrating ML concepts. Not intended for clinical use.(one day)
