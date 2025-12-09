import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

NUM_DAYS = 90
START_DATE = datetime(2024, 10, 1)

# Baseline values
BASE_SLEEP = 7.5
BASE_STEPS = 8000
BASE_WATER = 2000
BASE_MOOD = 4
BASE_STRESS = 2
BASE_PRODUCTIVITY = 6


def get_day_of_week_factor(date):
    """Returns adjustment factors based on weekday/weekend."""
    day = date.weekday()
    
    if day >= 5:  # Weekend
        return {
            'sleep': 1.1,
            'steps': 0.7,
            'stress': 0.6,
            'productivity': 0.3,
            'mood': 1.1
        }
    else:  # Weekday
        return {
            'sleep': 1.0,
            'steps': 1.0,
            'stress': 1.0,
            'productivity': 1.0,
            'mood': 1.0
        }


def simulate_stress_event(day_num):
    """Simulates predefined stress events (exam week, project deadline)."""
    if 20 <= day_num <= 25:
        return {
            'stress_add': 2.0,
            'sleep_mult': 0.75,
            'mood_mult': 0.7,
            'productivity_mult': 1.3
        }
    elif 60 <= day_num <= 73:
        return {
            'stress_add': 2.5,
            'sleep_mult': 0.65,
            'mood_mult': 0.65,
            'productivity_mult': 1.4
        }
    else:
        return {
            'stress_add': 0,
            'sleep_mult': 1.0,
            'mood_mult': 1.0,
            'productivity_mult': 1.0
        }


def add_realistic_noise(value, noise_percent=0.15):
    """Adds Gaussian noise to a value."""
    noise = np.random.normal(0, value * noise_percent)
    return value + noise


def simulate_sick_day(day_num):
    """Returns sick day modifiers for days 45-47."""
    if 45 <= day_num <= 47:
        return {
            'sleep_mult': 1.3,
            'steps_mult': 0.3,
            'mood_mult': 0.5,
            'stress_mult': 0.8,
            'productivity_mult': 0.2
        }
    return None


def generate_symptoms(sick, stress, mood):
    """Generates symptom strings based on health state."""
    if sick:
        symptoms_list = [
            "headache, fatigue",
            "sore throat, low energy",
            "congestion, feeling weak",
            "body aches, tired"
        ]
        return random.choice(symptoms_list)
    
    elif stress >= 4:
        symptoms_list = [
            "tension headache",
            "feeling anxious",
            "difficulty concentrating",
            "tired but can't sleep"
        ]
        return random.choice(symptoms_list)
    
    elif mood <= 2:
        symptoms_list = [
            "feeling down",
            "low motivation",
            "irritable",
            "unmotivated"
        ]
        return random.choice(symptoms_list)
    else:
        return "none"


# Data generation
def generate_health_data():
    """Generates synthetic health data with realistic patterns."""
    data = []
    
    for day_num in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day_num)
        
        dow_factor = get_day_of_week_factor(current_date)
        stress_event = simulate_stress_event(day_num)
        sick = simulate_sick_day(day_num)
        
        # Sleep calculation
        sleep = BASE_SLEEP * dow_factor['sleep'] * stress_event['sleep_mult']
        if sick:
            sleep *= sick['sleep_mult']
        sleep = add_realistic_noise(sleep, 0.12)
        sleep = max(4.0, min(12.0, sleep))
        
        # Steps calculation
        steps = BASE_STEPS * dow_factor['steps']
        if sick:
            steps *= sick['steps_mult']
        steps = add_realistic_noise(steps, 0.20)
        steps = max(1000, min(20000, int(steps)))
        
        # Water intake (correlated with activity)
        water = BASE_WATER + (steps - BASE_STEPS) * 0.05
        water = add_realistic_noise(water, 0.15)
        water = max(500, min(4000, int(water)))
        
        # Stress calculation
        stress = BASE_STRESS * dow_factor['stress'] + stress_event['stress_add']
        if sick:
            stress *= sick['stress_mult']
        stress = add_realistic_noise(stress, 0.20)
        stress = max(1, min(5, round(stress)))
        
        # Mood calculation with sleep and stress effects
        mood = BASE_MOOD * dow_factor['mood'] * stress_event['mood_mult']
        sleep_effect = (sleep - 6) * 0.3
        mood += sleep_effect
        stress_effect = (stress - BASE_STRESS) * -0.4
        mood += stress_effect
        if sick:
            mood *= sick['mood_mult']
        mood = add_realistic_noise(mood, 0.15)
        mood = max(1, min(5, round(mood)))
        
        # Productivity calculation
        productivity = BASE_PRODUCTIVITY * dow_factor['productivity'] * stress_event['productivity_mult']
        if sleep < 6:
            productivity *= 0.7
        if sick:
            productivity *= sick['productivity_mult']
        productivity = add_realistic_noise(productivity, 0.20)
        productivity = max(0, min(14, round(productivity, 1)))
        
        symptoms = generate_symptoms(sick, stress, mood)
        
        day_data = {
            'date': current_date.strftime('%Y-%m-%d'),
            'day_of_week': current_date.strftime('%A'),
            'sleep_hours': round(sleep, 1),
            'steps': steps,
            'water_ml': water,
            'mood': mood,
            'stress': stress,
            'productivity_hours': productivity,
            'symptoms': symptoms
        }
        
        data.append(day_data)
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    print("Generating synthetic health data...")
    print(f"Creating {NUM_DAYS} days of data starting from {START_DATE.date()}\n")
    
    df = generate_health_data()
    
    filename = "health_data_synthetic.csv"
    df.to_csv(filename, index=False)
    
    print(f"Data generated successfully")
    print(f"Saved to: {filename}\n")
    
    print("First 10 days of data:")
    print(df.head(10).to_string())
    
    print("\nDataset Statistics:")
    print(df.describe())
    
    print("\nSimulated Events:")
    print("  - Days 20-25: Exam week (high stress)")
    print("  - Days 45-47: Sick days")
    print("  - Days 60-73: Project deadline (very high stress)")
    
    print("\nData ready for model training.")