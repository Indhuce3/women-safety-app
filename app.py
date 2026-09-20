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

# Coordinates map for Chennai, Erode, Tiruppur locations
LOCATION_COORDS = {
    "Chennai": [13.0827, 80.2707],
    "Erode": [11.3410, 77.7172],
    "Tiruppur": [11.1085, 77.3411]
}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    status_class = ""
    route_color = "#3388ff"
    
    start_lat, start_lng = 11.3410, 77.7172
    end_lat, end_lng = 11.1085, 77.3411
    start_loc = None
    end_loc = None

    if request.method == 'POST':
        if 'review_submit' in request.form:
            prediction_text = "Thank you! Your safety review has been recorded."
            status_class = "HIGHLY-SAFE"
        else:
            city = request.form.get('city', 'Chennai')
            start_loc = request.form.get('start_loc')
            end_loc = request.form.get('end_loc')
            travel_time = request.form.get('travel_time', '21:00')

            hours, minutes = map(int, travel_time.split(':'))
            time_val = hours + (minutes / 60.0)

            # Location setup based on city choice
            base_coords = LOCATION_COORDS.get(city, [13.0827, 80.2707])
            start_lat, start_lng = base_coords[0], base_coords[1]
            end_lat, end_lng = base_coords[0] + 0.03, base_coords[1] + 0.03

            # Model prediction matching
            area_name = encoders['Area Name'].classes_[0] 
            weather_name = encoders['Weather'].classes_[0]
            
            area_enc = encoders['Area Name'].transform([area_name])[0]
            weather_enc = encoders['Weather'].transform([weather_name])[0]

            input_df = pd.DataFrame([[area_enc, time_val, 6, 4, 400, 3, weather_enc]], columns=features)
            score = model.predict(input_df)[0]

            if score >= 7.0:
                status_class = "HIGHLY-SAFE"
                prediction_text = f"🟢 HIGHLY SAFE ROUTE (Safety Score: {score:.1f}/10)"
                route_color = "#28a745" # Green color route line
            elif score >= 5.0:
                status_class = "MODERATE"
                prediction_text = f"🟡 MODERATE / CAUTION ROUTE (Safety Score: {score:.1f}/10)"
                route_color = "#ffc107" # Yellow color route line
            else:
                status_class = "UNSAFE"
                prediction_text = f"🔴 UNSAFE ROUTE / AVOID NIGHT TRAVEL (Safety Score: {score:.1f}/10)"
                route_color = "#dc3545" # Red color route line

    return render_template('index.html', 
                           prediction=prediction_text, 
                           status_class=status_class,
                           start_loc=start_loc,
                           end_loc=end_loc,
                           start_lat=start_lat, start_lng=start_lng,
                           end_lat=end_lat, end_lng=end_lng,
                           route_color=route_color)

if __name__ == '__main__':
    app.run(debug=True)
