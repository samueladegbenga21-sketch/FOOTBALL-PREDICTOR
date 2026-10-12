from flask import Flask, jsonify, request
import requests
import joblib
import numpy as np

app = Flask(__name__)

# Load your pre-trained machine learning model 
# (e.g., a model trained offline on historical match stats)
model = joblib.load('football_predictor_model.pkl')

# Set up your API credentials (e.g., API-Football)
API_KEY = 'YOUR_API_KEY'
BASE_URL = 'https://v3.football.api-sports.io'
HEADERS = {
    'x-apisports-key': API_KEY,
    'x-rapidapi-host': 'v3.football.api-sports.io'
}

@app.route('/predict_live', methods=['GET'])
def predict_live():
    # 1. Fetch live match data from the internet
    # Pass a live fixture ID (e.g., a Manchester United vs Chelsea match)
    fixture_id = request.args.get('fixture_id', '865912') 
    
    try:
        # Request live match statistics
        response = requests.get(f"{BASE_URL}/fixtures/statistics?fixture={fixture_id}", headers=HEADERS)
        data = response.json()
        
        # 2. Extract relevant live features for both teams
        home_stats = data['response'][0]['statistics']
        away_stats = data['response'][1]['statistics']
        
        # Helper function to grab specific stats safely
        def get_stat(stats_list, stat_name):
            val = next((item['value'] for item in stats_list if item['type'] == stat_name), 0)
            return float(val) if val is not None else 0.0

        # Extract high-value metrics like Expected Goals (xG)
        home_xg = get_stat(home_stats, 'expected_goals')
        away_xg = get_stat(away_stats, 'expected_goals')
        
        # 3. Format features for the model
        # You could also factor in live tactical shifts here if your model 
        # tracks whether a team is playing a narrow 4-3-1-2 diamond versus a wider 3-5-2
        live_features = np.array([[home_xg, away_xg]])
        
        # 4. Generate the prediction
        # Output probabilities for Home Win, Draw, or Away Win
        win_probabilities = model.predict_proba(live_features)[0]
        
        return jsonify({
            'fixture_id': fixture_id,
            'match_context': {
                'home_xg': home_xg,
                'away_xg': away_xg
            },
            'win_probabilities': {
                'home': round(win_probabilities[0], 2),
                'draw': round(win_probabilities[1], 2),
                'away': round(win_probabilities[2], 2)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f"Failed to generate prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Bind to 0.0.0.0 so it is ready to deploy to a cloud platform
    app.run(debug=True, host='0.0.0.0', port=5000)