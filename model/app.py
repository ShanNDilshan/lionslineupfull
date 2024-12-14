from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

# Load the Model and Encoder
with open("fitness_score_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("venue_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow Cross-Origin Requests

# Load data for team information
data = pd.read_excel("Modelnew.xlsx")
data = data[['Team', 'TeamID', 'Fitness_Score', 'Venue']]

# Define Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict_fitness_score():
    try:
        # Parse JSON data
        input_data = request.json
        print("Received data:", input_data)  # Log received data
        
        # Validate and normalize venue
        venue = input_data.get('venue', '').strip().replace(' ', '')  # Normalize venue
        if not venue:
            return jsonify({"error": "Please provide a valid venue"}), 400

        # Encode the venue
        try:
            venue_encoded = encoder.transform([venue])[0]
        except ValueError:
            return jsonify({"error": f"Invalid venue '{venue}' provided"}), 400

        # Predict fitness score
        prediction = model.predict([[venue_encoded]])
        predicted_score = prediction[0]

        # Find the team with the highest score
        highest_score_team = data[data['Fitness_Score'] == data['Fitness_Score'].max()]

        return jsonify({
            "predicted_fitness_score": predicted_score,
            "highest_score_team": highest_score_team['Team'].values[0]
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
