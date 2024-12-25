from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the model
model = joblib.load("../Models/crop_rec.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from the frontend
        data = request.json
        n = data.get('N')
        p = data.get('P')
        k = data.get('K')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        ph = data.get('ph')
        rainfall = data.get('rainfall')

        # Ensure all inputs are provided
        if None in [n, p, k, temperature, humidity, ph, rainfall]:
            return jsonify({'error': 'Missing input values'}), 400

        # Prepare input for the model
        input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])

        # Make prediction
        prediction = model.predict(input_data)
        return jsonify({'recommended_crop': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# Load the trained model
model_yield = joblib.load('../Models/crop_yield.pkl')  # Path to your trained model
scaler = joblib.load('../Models/scaler.pkl')  # Path to your scaler (MinMaxScaler)
label_encoder_crop = joblib.load('../Models/label_encoder_crop.pkl')  # Path to LabelEncoder for Crop
label_encoder_season = joblib.load('../Models/label_encoder_season.pkl')  # Path to LabelEncoder for Season

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        crop = data.get('crop')
        season = data.get('season')
        area = data.get('area')
        annual_rainfall = data.get('annual_rainfall')
        fertilizer = data.get('fertilizer')
        pesticide = data.get('pesticide')

        # Encode the crop and season inputs
        crop_encoded = label_encoder_crop.transform([crop])[0]
        season_encoded = label_encoder_season.transform([season])[0]

        # Create the feature vector (same structure as used in training)
        features = np.array([[crop_encoded, season_encoded, area, annual_rainfall, fertilizer, pesticide]])

        # Normalize the features
        features_normalized = scaler.transform(features)

        # Predict the crop yield
        predicted_yield = model_yield.predict(features_normalized)

        # Return the result
        return jsonify({'predicted_yield': round(predicted_yield[0], 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
