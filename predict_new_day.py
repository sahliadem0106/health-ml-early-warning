import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Load models
print("Loading trained models...")

# Load Anomaly Detection Model
try:
    with open('anomaly_model.pkl', 'rb') as f:
        anomaly_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        anomaly_scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        anomaly_features = pickle.load(f)
    print("Anomaly detection model loaded")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run 'python train_anomaly_model.py' first")
    exit(1)

# Load Mood Prediction Model
try:
    with open('mood_prediction_model.pkl', 'rb') as f:
        mood_model = pickle.load(f)
    with open('mood_scaler.pkl', 'rb') as f:
        mood_scaler = pickle.load(f)
    with open('mood_feature_columns.pkl', 'rb') as f:
        mood_features = pickle.load(f)
    print("Mood prediction model loaded")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run 'python train_prediction_model.py' first")
    exit(1)

print()

# Load training data for score normalization
try:
    df_train = pd.read_csv('health_data_with_predictions.csv')
    min_score = df_train['anomaly_score'].min()
    max_score = df_train['anomaly_score'].max()
    print("Training data loaded for calibration\n")
except FileNotFoundError:
    print("Warning: Could not load training data, using default score range\n")
    min_score = -0.5
    max_score = 0.5


def prepare_anomaly_features(data):
    """
    Prepares features for anomaly detection model.
    
    Args:
        data: dict with today's health metrics
    
    Returns:
        DataFrame ready for anomaly model
    """
    features = {
        'sleep_hours': data['sleep_hours'],
        'steps': data['steps'],
        'water_ml': data['water_ml'],
        'mood': data['mood'],
        'stress': data['stress'],
        'productivity_hours': data['productivity_hours'],
        'sleep_3day_avg': data.get('sleep_3day_avg', data['sleep_hours']),
        'stress_3day_avg': data.get('stress_3day_avg', data['stress']),
        'mood_3day_avg': data.get('mood_3day_avg', data['mood']),
        'sleep_change': data.get('sleep_change', 0),
        'stress_change': data.get('stress_change', 0),
        'mood_change': data.get('mood_change', 0)
    }
    
    df = pd.DataFrame([features])[anomaly_features]
    return df


def prepare_mood_features(data):
    """
    Prepares features for mood prediction model.
    
    Args:
        data: dict with today's health metrics
    
    Returns:
        DataFrame ready for mood model
    """
    features = {
        'sleep_hours': data['sleep_hours'],
        'steps': data['steps'],
        'water_ml': data['water_ml'],
        'mood': data['mood'],
        'stress': data['stress'],
        'productivity_hours': data['productivity_hours'],
        'sleep_avg_3d': data.get('sleep_avg_3d', data['sleep_hours']),
        'sleep_avg_7d': data.get('sleep_avg_7d', data['sleep_hours']),
        'stress_avg_3d': data.get('stress_avg_3d', data['stress']),
        'mood_avg_3d': data.get('mood_avg_3d', data['mood']),
        'steps_avg_3d': data.get('steps_avg_3d', data['steps']),
        'sleep_change': data.get('sleep_change', 0),
        'stress_change': data.get('stress_change', 0),
        'mood_change': data.get('mood_change', 0),
        'steps_change': data.get('steps_change', 0),
        'sleep_std_7d': data.get('sleep_std_7d', 0.5),
        'stress_std_3d': data.get('stress_std_3d', 0.5),
        'is_weekend': data.get('is_weekend', 0),
        'sleep_stress_interaction': data['sleep_hours'] * data['stress'],
        'productivity_per_step': data['productivity_hours'] / (data['steps'] + 1),
        'cumulative_stress': data.get('cumulative_stress', data['stress'] * 7)
    }
    
    df = pd.DataFrame([features])[mood_features]
    return df


def get_anomaly_prediction(data):
    """
    Predicts if today is an anomaly.
    
    Returns:
        dict with anomaly results
    """
    X = prepare_anomaly_features(data)
    X_scaled = anomaly_scaler.transform(X)
    
    prediction = anomaly_model.predict(X_scaled)[0]
    score = anomaly_model.decision_function(X_scaled)[0]
    
    risk_score = 100 * (1 - (score - min_score) / (max_score - min_score))
    risk_score = np.clip(risk_score, 0, 100)
    
    return {
        'is_anomaly': prediction == -1,
        'anomaly_score': float(score),
        'risk_score': float(risk_score)
    }


def get_mood_prediction(data):
    """
    Predicts tomorrow's mood.
    
    Returns:
        dict with mood prediction
    """
    X = prepare_mood_features(data)
    X_scaled = mood_scaler.transform(X)
    
    predicted_mood = mood_model.predict(X_scaled)[0]
    predicted_mood = np.clip(predicted_mood, 1, 5)
    
    return {
        'predicted_mood': float(predicted_mood),
        'confidence': 'high' if 2 <= predicted_mood <= 4 else 'medium'
    }


def generate_recommendations(data, anomaly_result, mood_result):
    """
    Generates personalized recommendations based on predictions.
    
    Returns:
        list of recommendation strings
    """
    recommendations = []
    
    risk = anomaly_result['risk_score']
    predicted_mood = mood_result['predicted_mood']
    
    # High risk recommendations
    if risk > 75:
        recommendations.append("HIGH RISK DETECTED - Immediate attention needed")
        
        if data['sleep_hours'] < 6:
            recommendations.append("URGENT: Get 8+ hours of sleep tonight")
        
        if data['stress'] >= 4:
            recommendations.append("URGENT: Practice stress management (meditation, deep breathing)")
        
        if data['mood'] <= 2:
            recommendations.append("Consider reaching out to a friend or counselor")
            
        if data['steps'] < 3000:
            recommendations.append("Try to get some light physical activity")
    
    # Medium risk recommendations
    elif risk > 50:
        recommendations.append("Moderate risk - Monitor your health closely")
        
        if data['sleep_hours'] < 7:
            recommendations.append("Aim for 7-8 hours of sleep tonight")
        
        if data['stress'] >= 3:
            recommendations.append("Take breaks throughout the day")
        
        if data['water_ml'] < 1500:
            recommendations.append("Increase water intake (target: 2L)")
    
    # Low risk but poor predicted mood
    elif predicted_mood < 2.5:
        recommendations.append("Tomorrow's mood outlook is concerning")
        recommendations.append("Prioritize good sleep tonight (7-8 hours)")
        recommendations.append("Consider a lighter schedule tomorrow")
    
    # Good outlook
    elif predicted_mood >= 4:
        recommendations.append("Great outlook for tomorrow")
        recommendations.append("Keep up your healthy habits")
    
    # General recommendations
    if data['productivity_hours'] > 10:
        recommendations.append("You're working too much - risk of burnout")
    
    if data['steps'] < 5000:
        recommendations.append("Try to reach 8,000 steps daily")
    
    if not recommendations:
        recommendations.append("Everything looks normal - maintain current habits")
    
    return recommendations


def format_output(data, anomaly_result, mood_result, recommendations):
    """
    Formats and prints the prediction report.
    """
    print("=" * 70)
    print("           HEALTH PREDICTION REPORT")
    print("=" * 70)
    print()
    
    # Today's Data Summary
    print("TODAY'S DATA:")
    print("-" * 70)
    print(f"   Sleep:        {data['sleep_hours']} hours")
    print(f"   Steps:        {data['steps']:,}")
    print(f"   Water:        {data['water_ml']} ml")
    print(f"   Mood:         {data['mood']}/5")
    print(f"   Stress:       {data['stress']}/5")
    print(f"   Productivity: {data['productivity_hours']} hours")
    print()
    
    # Anomaly Detection Results
    print("ANOMALY DETECTION:")
    print("-" * 70)
    
    risk = anomaly_result['risk_score']
    if risk >= 75:
        risk_level = "HIGH RISK"
    elif risk >= 50:
        risk_level = "MEDIUM RISK"
    elif risk >= 25:
        risk_level = "LOW RISK"
    else:
        risk_level = "VERY LOW RISK"
    
    print(f"   Status:      {risk_level}")
    print(f"   Risk Score:  {risk:.1f}/100")
    
    bar_length = int(risk / 2)
    bar = "#" * bar_length
    print(f"   Risk Bar:    [{bar:<50}] {risk:.0f}%")
    print()
    
    # Mood Prediction Results
    print("TOMORROW'S MOOD PREDICTION:")
    print("-" * 70)
    predicted_mood = mood_result['predicted_mood']
    
    if predicted_mood >= 4:
        mood_desc = "Good"
    elif predicted_mood >= 3:
        mood_desc = "Neutral"
    elif predicted_mood >= 2:
        mood_desc = "Concerning"
    else:
        mood_desc = "Poor"
    
    print(f"   Predicted Mood: {predicted_mood:.1f}/5.0")
    print(f"   Outlook:        {mood_desc}")
    print(f"   Confidence:     {mood_result['confidence'].title()}")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 70)
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    print()
    print("=" * 70)


def get_user_input():
    """
    Gets health data from user via command line.
    """
    print("=" * 70)
    print("           ENTER TODAY'S HEALTH DATA")
    print("=" * 70)
    print()
    
    try:
        sleep_hours = float(input("Sleep hours (e.g., 7.5): "))
        steps = int(input("Steps taken (e.g., 8000): "))
        water_ml = int(input("Water intake in ml (e.g., 2000): "))
        mood = int(input("Mood 1-5 (1=worst, 5=best): "))
        stress = int(input("Stress 1-5 (1=lowest, 5=highest): "))
        productivity_hours = float(input("Productivity hours (e.g., 6): "))
        
        if not (4 <= sleep_hours <= 12):
            print("Warning: Sleep hours should be between 4-12")
        if not (1 <= mood <= 5):
            print("Warning: Mood should be between 1-5")
            mood = max(1, min(5, mood))
        if not (1 <= stress <= 5):
            print("Warning: Stress should be between 1-5")
            stress = max(1, min(5, stress))
        
        print()
        
        data = {
            'sleep_hours': sleep_hours,
            'steps': steps,
            'water_ml': water_ml,
            'mood': mood,
            'stress': stress,
            'productivity_hours': productivity_hours
        }
        
        return data
        
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return None
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        exit(0)


def run_prediction(data):
    """
    Runs both predictions and displays results.
    """
    print("Analyzing your data...\n")
    
    anomaly_result = get_anomaly_prediction(data)
    mood_result = get_mood_prediction(data)
    recommendations = generate_recommendations(data, anomaly_result, mood_result)
    
    format_output(data, anomaly_result, mood_result, recommendations)


def run_demo():
    """
    Runs demo with pre-defined scenarios.
    """
    print("=" * 70)
    print("           RUNNING DEMO MODE")
    print("=" * 70)
    print("\nTesting 3 scenarios: Healthy, Stressed, Exhausted\n")
    
    scenarios = [
        {
            'name': 'HEALTHY DAY',
            'data': {
                'sleep_hours': 8.0,
                'steps': 10000,
                'water_ml': 2500,
                'mood': 5,
                'stress': 1,
                'productivity_hours': 7
            }
        },
        {
            'name': 'STRESSED DAY',
            'data': {
                'sleep_hours': 6.0,
                'steps': 5000,
                'water_ml': 1200,
                'mood': 3,
                'stress': 4,
                'productivity_hours': 9
            }
        },
        {
            'name': 'EXHAUSTED DAY',
            'data': {
                'sleep_hours': 4.5,
                'steps': 2000,
                'water_ml': 800,
                'mood': 2,
                'stress': 5,
                'productivity_hours': 3
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"  SCENARIO {i}/3: {scenario['name']}")
        print('='*70)
        
        anomaly_result = get_anomaly_prediction(scenario['data'])
        mood_result = get_mood_prediction(scenario['data'])
        recommendations = generate_recommendations(scenario['data'], anomaly_result, mood_result)
        
        format_output(scenario['data'], anomaly_result, mood_result, recommendations)
        
        if i < len(scenarios):
            input("\nPress Enter to continue to next scenario...")


if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print()
    print("     HEALTH EARLY WARNING SYSTEM - PREDICTION ENGINE")
    print()
    print("=" * 70)
    print("\n")
    
    print("Choose mode:")
    print("  1. Interactive Mode (enter your data)")
    print("  2. Demo Mode (see pre-defined scenarios)")
    print("  3. Exit")
    print()
    
    try:
        choice = input("Enter choice (1/2/3): ").strip()
        print()
        
        if choice == '1':
            while True:
                data = get_user_input()
                
                if data is None:
                    retry = input("\nTry again? (y/n): ").lower()
                    if retry != 'y':
                        break
                    continue
                
                run_prediction(data)
                
                print()
                again = input("Predict another day? (y/n): ").lower()
                if again != 'y':
                    print("\nThank you for using the Health Prediction System!")
                    break
                print("\n")
        
        elif choice == '2':
            run_demo()
            print("\n\nDemo complete. Run in Interactive Mode (option 1) to test your own data.")
        
        elif choice == '3':
            print("Goodbye!")
        
        else:
            print("Invalid choice. Please run again and choose 1, 2, or 3.")
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you've run the training scripts first.")