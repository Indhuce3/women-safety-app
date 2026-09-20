from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Model-ஐ Load செய்தல்
with open('safety_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    encoders = data['encoders']
    features = data['features']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    
    if request.method == 'POST':
        try:
            area = request.form['area']
            time = float(request.form['time'])
            lights = int(request.form['lights'])
            patrol = int(request.form['patrol'])
            distance = int(request.form['distance'])
            crime = int(request.form['crime'])
            weather = request.form['weather']

            # Encoding Input
            area_enc = encoders['Area Name'].transform([area])[0]
            weather_enc = encoders['Weather'].transform([weather])[0]

            input_df = pd.DataFrame([[area_enc, time, lights, patrol, distance, crime, weather_enc]], columns=features)
            score = model.predict(input_df)[0]
            
            if score >= 7.5:
                status = "🟢 HIGHLY SAFE"
            elif score >= 5.0:
                status = "🟡 CAUTION / MODERATE"
            else:
                status = "🔴 UNSAFE / AVOID ROUTE"
                
            prediction_text = f"Safety Score: ⭐ {score:.2f} / 10 [{status}]"
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    locations = list(encoders['Area Name'].classes_)
    weathers = list(encoders['Weather'].classes_)
    return render_template('index.html', prediction=prediction_text, locations=locations, weathers=weathers)

if __name__ == '__main__':
    app.run(debug=True)