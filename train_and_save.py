import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Dataset Generation
np.random.seed(42)
n_routes = 300
locations = ['T-Nagar', 'Adyar', 'Tambaram', 'Velachery', 'Central-Zone', 'Anna-Nagar', 'Guindy', 'Porur']
weather_conditions = ['Clear', 'Rainy', 'Foggy']

raw_data = {
    'Route ID': [f'Route_{i}' for i in range(n_routes)],
    'Area Name': np.random.choice(locations, n_routes),
    'Time of Night (24hr)': np.random.uniform(20.0, 28.0, n_routes),
    'Streetlight Density Score (1-10)': np.random.randint(1, 11, n_routes),
    'Police Patrol Frequency (per night)': np.random.randint(0, 6, n_routes),
    'Distance to Public Hotspot (meters)': np.random.randint(50, 2000, n_routes),
    'Historical Crime Rate (Last 1 Year)': np.random.randint(0, 15, n_routes),
    'Weather': np.random.choice(weather_conditions, n_routes)
}

df = pd.DataFrame(raw_data)

base_safety = 5.0 + (df['Streetlight Density Score (1-10)'] * 0.3) + (df['Police Patrol Frequency (per night)'] * 0.4) \
              - (df['Historical Crime Rate (Last 1 Year)'] * 0.2) - ((df['Time of Night (24hr)'] - 20) * 0.15) \
              - (df['Distance to Public Hotspot (meters)'] / 1000 * 0.5)

df['Safety Index Score'] = base_safety + np.random.normal(0, 0.3, n_routes)
df['Safety Index Score'] = df['Safety Index Score'].clip(1.0, 10.0).round(1)

# Cleaning
df = df.dropna(subset=['Safety Index Score'])
df['Streetlight Density Score (1-10)'] = df['Streetlight Density Score (1-10)'].fillna(df['Streetlight Density Score (1-10)'].median())

# Encoding
features = ['Area Name', 'Time of Night (24hr)', 'Streetlight Density Score (1-10)', 
            'Police Patrol Frequency (per night)', 'Distance to Public Hotspot (meters)', 
            'Historical Crime Rate (Last 1 Year)', 'Weather']

X = df[features].copy()
y = df['Safety Index Score'].copy()

encoders = {}
for col in ['Area Name', 'Weather']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Model-ஐ சேமித்தல்
model_data = {
    'model': model,
    'encoders': encoders,
    'features': features
}

with open('safety_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ safety_model.pkl file உருவாக்கப்பட்டுவிட்டது!")