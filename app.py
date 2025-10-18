from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_classifier_model.joblib')
model = joblib.load(model_path)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Heart Disease Classifier API is running'})

@app.route('/predict', methods=['POST'])
def predict():
  try:
    data = request.get_json(force=True)
    
    if 'features' not in data:
      return jsonify({'error': 'No features provided'}), 400
    
    X = pd.DataFrame(data['features'], index=[0])
    no_probability, yes_probability = model.predict_proba(X).tolist()[0]
    prediction = model.predict(X).tolist()
    
    return jsonify({
      'prediction': prediction[0] == 1,
      'no_probability': round(no_probability, 4),
      'yes_probability': round(yes_probability, 4)
    })
  except Exception as e:
    return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port, debug=False)