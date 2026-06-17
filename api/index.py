from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

# Tell Flask to look in its current directory (api/) for static files
app = Flask(__name__, static_folder='.')

# 1. Load the pre-trained brains
ct = joblib.load('api/fish_transformer.joblib')
scaler = joblib.load('api/fish_scaler.joblib')
model = joblib.load('api/fish_model.joblib')

# 2. NEW HOMEPAGE ROUTE: This serves your index.html when someone opens the link!
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

# 3. Your existing prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        user_input = pd.DataFrame([[
            data['species'], 
            data['length1'], 
            data['length2'], 
            data['length3'], 
            data['height'], 
            data['width']
        ]], columns=['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width'])
        
        encoded_data = ct.transform(user_input)
        scaled_data = scaler.transform(encoded_data)
        prediction = model.predict(scaled_data)
        
        return jsonify({'predicted_weight': round(prediction[0], 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
