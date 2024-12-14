from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = joblib.load('model_XGBoost.pkl')

# Player encoding dictionary
player_encoding = {
    'Akila Dananjaya': 0, 'Angelo Mathews': 1, 'Ashen Bandara': 2, 'Asitha Fernando': 3,
    'Avishka Fernando': 4, 'Bhanuka Rajapaksa': 5, 'Binura Fernando': 6, 'Chamika Karunaratne': 7,
    'Charith Asalanka': 8, 'Dasun Shanaka': 9, 'Dhananjaya Lakshan': 10, 'Dhananjaya de Silva': 11,
    'Dilshan Madushanka': 12, 'Dilshan Munaweera': 13, 'Dinesh Chandimal': 14, 'Dunith Wellalage': 15,
    'Dushmantha Chameera': 16, 'Duvindu Tillakaratne': 17, 'Janith Liyanage': 18, 'Jeffrey Vandersay': 19,
    'Kamindu Mendis': 20, 'Kasun Rajitha': 21, 'Kusal Mendis': 22, 'Kusal Perera': 23,
    'Lahiru Kumara': 24, 'Lahiru Madushanka': 25, 'Lahiru Udara': 26, 'Lakshan Sandakan': 27,
    'Lasith Croospulle': 28, 'Maheesh Theekshana': 29, 'Matheesha Pathirana': 30, 'Minod Bhanuka': 31,
    'Nishan Madushka': 32, 'Nuwan Thushara': 33, 'Nuwanidu Fernando': 34, 'Oshada Fernando': 35,
    'Pathum Nissanka': 36, 'Pramod Madushan': 37, 'Ramesh Mendis': 38, 'Sadeera Samarawickrama': 39,
    'Sahan Arachchige': 40, 'Vijayakanth Viyaskanth': 41, 'Wanindu Hasaranga': 42
}

# Function to make the prediction
def predict_favoritism_score(lineup):
    # Filter out any empty selections
    lineup = [player for player in lineup if player]

    # Check for duplicate players
    if len(set(lineup)) != len(lineup):
        return None  # Duplicate players found
    
    # Encode the selected players
    try:
        encoded_lineup = [player_encoding[player] for player in lineup]
    except KeyError:
        return None  # If an invalid player is selected, return None
    
    encoded_lineup_df = pd.DataFrame([encoded_lineup], columns=[f'Player_{i + 1}' for i in range(len(encoded_lineup))])
    
    # Make the prediction
    score = model.predict(encoded_lineup_df)
    return score[0]


@app.route('/api/pred', methods=['POST'])
def predict_api():
    """
    API endpoint to predict favoritism score.
    Expects a JSON payload with 'players' as a list of 11 player names.
    """
    data = request.get_json()

    # Ensure the 'players' key exists in the JSON payload
    if not data or 'players' not in data:
        return jsonify({'error': 'Invalid request. Please provide a list of players.'}), 400

    players = data['players']

    # Validate that exactly 11 players are provided
    if len(players) != 11:
        return jsonify({'error': 'Exactly 11 players must be provided.'}), 400

    # Ensure there are no duplicate players
    if len(set(players)) != len(players):
        return jsonify({'error': 'Duplicate players are not allowed.'}), 400

    # Predict the favoritism score
    predicted_score = predict_favoritism_score(players)
    
    if predicted_score is not None:
        # Convert the score to percentage
        percentage = (predicted_score / 7035) * 100  # Assuming the max score is 7035
        
        # Ensure float32 is converted to Python float
        return jsonify({
          #  'players': players,
            'predicted_score': float(predicted_score),  # Convert here
            'percentage': round(float(percentage), 2)  # And here
        }), 200
    else:
        return jsonify({'error': 'Invalid player selection. Please ensure all players are valid.'}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8000)