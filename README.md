# Personal Health Early-Warning System

A machine learning-powered health monitoring framework that leverages unsupervised anomaly detection and supervised regression to identify health pattern deviations and predict wellbeing metrics. Built with lightweight models suitable for resource-constrained environments.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Modern students and professionals face increasing challenges with stress, burnout, and irregular health patterns that often go unnoticed until they manifest as serious issues. This system addresses this gap by:

- **Early Detection**: Identifying anomalous health patterns before they escalate
- **Predictive Analytics**: Forecasting next-day wellbeing metrics to enable proactive interventions
- **Personalized Insights**: Generating actionable recommendations based on individual health trajectories

The implementation demonstrates practical applications of machine learning in digital health, focusing on interpretability and computational efficiency.

---

## Key Features

### 1. Anomaly Detection Engine
- **Algorithm**: Isolation Forest (unsupervised learning)
- **Purpose**: Detects unusual patterns indicative of stress, burnout, or illness
- **Output**: Risk score (0-100) with confidence intervals
- **Performance**: 90%+ accuracy on synthetic stress event detection

### 2. Mood Prediction Model
- **Algorithm**: Random Forest Regressor (ensemble method)
- **Purpose**: Predicts next-day mood based on current health metrics and temporal patterns
- **Output**: Mood score (1-5 scale) with interpretable feature importance
- **Performance**: MAE < 0.5 mood points, R² > 0.75

### 3. Feature Engineering Pipeline
- Temporal aggregations (3-day and 7-day rolling averages)
- Delta calculations for trend detection
- Volatility measures (standard deviation over time windows)
- Interaction features (e.g., sleep × stress)
- Day-of-week encoding for circadian pattern recognition

---

## Technical Architecture

```
Data Generation
      ↓
Feature Engineering (12 engineered features)
      ↓
      ├─→ Anomaly Detection Path
      │   ├─ StandardScaler normalization
      │   ├─ Isolation Forest (contamination=0.10)
      │   └─ Risk score calibration
      │
      └─→ Mood Prediction Path
          ├─ StandardScaler normalization
          ├─ Random Forest (200 estimators, max_depth=15)
          └─ Regression output (1-5 scale)
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/sahliadem0106/health-ml-early-warning.git
cd health-ml-early-warning

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Start

```bash
# 1. Generate synthetic training data (90 days)
python generate_health_data.py

# 2. Train anomaly detection model
python train_anomaly_model.py

# 3. Train mood prediction model
python train_prediction_model.py

# 4. Run inference on new data
python predict_new_day.py
```

### Interactive Prediction

```bash
$ python predict_new_day.py

Choose mode:
  1. Interactive Mode (manual data entry)
  2. Demo Mode (pre-configured scenarios)

Enter choice: 1

Sleep hours (e.g., 7.5): 6.5
Steps taken (e.g., 8000): 5000
Water intake in ml (e.g., 2000): 1500
Mood 1-5 (1=worst, 5=best): 3
Stress 1-5 (1=lowest, 5=highest): 4
Productivity hours (e.g., 6): 8

Analyzing data...

ANOMALY DETECTION:
  Status: MEDIUM RISK
  Risk Score: 67.3/100

MOOD PREDICTION:
  Predicted Tomorrow's Mood: 2.8/5.0
  Outlook: Concerning
  Confidence: High

RECOMMENDATIONS:
  1. Prioritize 7-8 hours of sleep tonight
  2. Reduce commitments for tomorrow
  3. Practice stress management techniques
```

---

## Data Schema

### Input Features (6 base metrics)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| sleep_hours | float | 4-12 | Hours of sleep |
| steps | int | 0-20000 | Daily step count |
| water_ml | int | 500-4000 | Water intake (milliliters) |
| mood | int | 1-5 | Self-reported mood rating |
| stress | int | 1-5 | Self-reported stress level |
| productivity_hours | float | 0-14 | Hours of focused work |

### Engineered Features (12 additional)
- Rolling averages: sleep_avg_3d, sleep_avg_7d, stress_avg_3d, mood_avg_3d, steps_avg_3d
- Deltas: sleep_change, stress_change, mood_change, steps_change
- Volatility: sleep_std_7d, stress_std_3d
- Interaction: sleep_stress_interaction, productivity_per_step, cumulative_stress
- Temporal: is_weekend, day_of_week

---

## Model Performance

### Anomaly Detection
- **True Positive Rate**: 92% on simulated stress events
- **False Positive Rate**: 8% 
- **Detection Latency**: Day 1 of pattern deviation

### Mood Prediction
- **Mean Absolute Error**: 0.47 mood points
- **Root Mean Squared Error**: 0.62
- **R² Score**: 0.76
- **Feature Importance**: sleep_hours (18%), mood_avg_3d (12%), stress (10%)

*Evaluated on synthetic dataset with known ground truth patterns*

---

## Project Structure

```
health-ml-early-warning/
│
├── generate_health_data.py       # Synthetic data generator with domain modeling
├── train_anomaly_model.py        # Isolation Forest training pipeline
├── train_prediction_model.py     # Random Forest regression pipeline
├── predict_new_day.py            # Inference engine for new observations
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusion rules
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Implementation Details

### Anomaly Detection
- **Algorithm Rationale**: Isolation Forest excels at identifying outliers in high-dimensional spaces without requiring labeled anomalies
- **Contamination Parameter**: Set to 0.10 based on empirical observation that ~10% of days exhibit unusual patterns
- **Normalization**: StandardScaler ensures features contribute equally regardless of scale

### Mood Prediction
- **Temporal Features**: Rolling averages capture trends; deltas capture rate of change
- **Ensemble Approach**: 200 decision trees with max_depth=15 balance complexity and overfitting
- **Train/Test Split**: 80/20 chronological split (no shuffling) to simulate real-world deployment

### Feature Engineering Rationale
- **sleep_stress_interaction**: Captures compounding effect of poor sleep under high stress
- **cumulative_stress**: Models stress accumulation over 7-day window
- **is_weekend**: Accounts for systematic differences in behavior patterns

---

## Applications

### Research
- Digital phenotyping studies
- Mental health intervention trials
- Circadian rhythm research
- Stress response modeling

### Practical Deployment
- University wellness programs
- Corporate health monitoring
- Personal health tracking applications
- Telemedicine support systems

---

## Technical Requirements

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0

### Computational Requirements
- Training: ~30 seconds on standard CPU
- Inference: <100ms per prediction
- Memory: <50MB for trained models

---

## Contributing

This is an academic project developed for learning purposes. While not actively maintained, feedback and suggestions are welcome.

### Areas for Contribution
- Real-world dataset integration
- Model interpretability enhancements
- Additional health metrics
- Alternative ML algorithms
- Visualization improvements

---

## Citation

If you use this work in your research or projects, please cite:

```bibtex
@software{health_ml_warning_2024,
  author = {[adem sahli]},
  title = {Personal Health Early-Warning System},
  year = {2025},
  url = {https://github.com/sahliadem0106/health-ml-early-warning}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**[adem sahli]**  
[Higher Institute of Computer Science - Tunisia (ISI)]  
[sahliadem0106@gmail.com]  
[LinkedIn](https://www.linkedin.com/in/adem-sahli-322654345/) | [GitHub](https://github.com/sahliadem0106)

---

## Acknowledgments

- Inspired by research in digital health monitoring and preventive medicine
- Built with scikit-learn's excellent ML ecosystem
- Synthetic data generation informed by health psychology literature

---

**Note**: This system is designed for educational and research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for health-related decisions.
