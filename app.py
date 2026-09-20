from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'safety_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    encoders = data['encoders']
    features = data['features']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    status_class = ""
    
    locations = list(encoders['Area Name'].classes_)

    if request.method == 'POST':
        try:
            start_area = request.form['start_area']
            end_area = request.form['end_area']
            travel_time_str = request.form['travel_time']
            
            # Convert HH:MM time to float format (e.g. 22:30 -> 22.5)
            hours, minutes = map(int, travel_time_str.split(':'))
            time_val = hours + (minutes / 60.0)
            if time_val < 12:  # Treat early morning as night continuation
                time_val += 24.0

            # Default smart automated backend values
            lights = 7
            patrol = 5
            distance = 300
            crime = 2
            weather = 'Clear'

            area_enc = encoders['Area Name'].transform([start_area])[0]
            weather_enc = encoders['Weather'].transform([weather])[0]

            input_df = pd.DataFrame([[area_enc, time_val, lights, patrol, distance, crime, weather_enc]], columns=features)
            score = model.predict(input_df)[0]

            if score >= 7.0:
                status = "HIGHLY SAFE ROUTE"
                status_class = "safe"
            elif score >= 5.0:
                status = "MODERATE / USE CAUTION"
                status_class = "caution"
            else:
                status = "UNSAFE ROUTE / AVOID"
                status_class = "unsafe"

            prediction_text = f"Safety Score: ⭐ {score:.1f} / 10 ({status})"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"
            status_class = "unsafe"

    return render_template('index.html', prediction=prediction_text, status_class=status_class, locations=locations)

if __name__ == '__main__':
    app.run(debug=True)
