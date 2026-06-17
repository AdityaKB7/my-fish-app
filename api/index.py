from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load the pre-trained brains (DO NOT train anything here!)
ct = joblib.load('api/fish_transformer.joblib')
scaler = joblib.load('api/fish_scaler.joblib')
model = joblib.load('api/fish_model.joblib')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 2. Package the incoming website data exactly how Colab expects it
        # We use a Pandas DataFrame so the ColumnTransformer recognizes the 'Species' column
        user_input = pd.DataFrame([[
            data['species'], 
            data['length1'], 
            data['length2'], 
            data['length3'], 
            data['height'], 
            data['width']
        ]], columns=['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width'])
        
        # 3. Run the data through your pipeline
        encoded_data = ct.transform(user_input)     # Turns text to binary numbers
        scaled_data = scaler.transform(encoded_data) # Squishes the numbers to scale
        
        # 4. Make the final prediction
        prediction = model.predict(scaled_data)
        
        return jsonify({'predicted_weight': round(prediction[0], 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)