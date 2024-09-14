

import requests

def fetch_prediction(data):
    url = 'http://127.0.0.1:5000/predict'
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        prediction = response.json()
        return prediction
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

if _name_ == "_main_":
    # Example of dynamic input collection
    symptoms = [
        'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
        'Sore Throat', 'Rash', 'Headache', 'Nausea',
        'Vomiting', 'Diarrhea', 'Muscle Pain'
    ]

# Example symptom data
data = {}

for symptom in symptoms:
        value = int(input(f"Enter value for {symptom} (0 or 1): "))
        data[symptom] = value

# Fetch prediction
prediction = fetch_prediction(data)
if prediction:
    print(f"Random Forest Prediction: {prediction['RandomForestPrediction']}")
    print(f"Gradient Boosting Prediction: {prediction['GradientBoostingPrediction']}")
