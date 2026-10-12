from flask import Flask, jsonify, request
import requests
import numpy as np
from collections import Counter

app = Flask(__name__)

API_KEY = 'YOUR_API_KEY'
BASE_URL = 'https://v3.football.api-sports.io'
HEADERS = {
    'x-apisports-key': API_KEY,
    'x-rapidapi-host': 'v3.football.api-sports.io'
}

def simulate_match(home_xg, away_xg, num_simulations=10000):
    """
    Runs a Monte Carlo simulation using Poisson distributions.
    The xG acts as 'lambda' (the expected rate of occurrence) for the Poisson equation.
    """
    # 1. Generate an array of 10,000 possible goal counts for each team
    home_simulated_goals = np.random.poisson(lam=home_xg, size=num_simulations)
    away_simulated_goals = np.random.poisson(lam=away_xg, size=num_simulations)
    
    # 2. Combine arrays into exact scoreline strings (e.g., "2-1")
    scorelines = [f"{h}-{a}" for h, a in zip(home_simulated_goals, away_simulated_goals)]
    
    # 3. Tally up the most frequent scorelines
    score_counts = Counter(scorelines)
    
    # 4. Calculate percentage probabilities for the top 5 exact scores
    top_scores = {}
    for score, count in score_counts.most_common(5):
        top_scores[score] = round((count / num_simulations) * 100, 2)
        
    # 5. Calculate overall 1X2 match odds based on the 10,000 outcomes
    home_wins = sum(1 for h, a in zip(home_simulated_goals, away_simulated_goals) if h > a)
    draws = sum(1 for h, a in zip(home_simulated_goals, away_simulated_goals) if h == a)
    away_wins = sum(1 for h, a in zip(home_simulated_goals, away_simulated_goals) if h < a)
    
    match_odds = {
        'home_win_pct': round((home_wins / num_simulations) * 100, 2),
        'draw_pct': round((draws / num_simulations) * 100, 2),
        'away_win_pct': round((away_wins / num_simulations) * 100, 2)
    }
    
    return top_scores, match_odds

@app.route('/predict_live', methods=['GET'])
def predict_live():
    fixture_id = request.args.get('fixture_id', '865912') 
    
    try:
        # Fetch live match statistics
        response = requests.get(f"{BASE_URL}/fixtures/statistics?fixture={fixture_id}", headers=HEADERS)
        data = response.json()
        
        home_stats = data['response'][0]['statistics']
        away_stats = data['response'][1]['statistics']
        
        def get_stat(stats_list, stat_name):
            val = next((item['value'] for item in stats_list if item['type'] == stat_name), 0)
            return float(val) if val is not None else 0.0

        # Extract live Expected Goals (xG)
        home_xg = get_stat(home_stats, 'expected_goals')
        away_xg = get_stat(away_stats, 'expected_goals')
        
        # Fallback safeguard: if match just started and xG is 0, give a tiny baseline
        # to prevent math errors in the distribution
        home_xg = home_xg if home_xg > 0 else 0.1
        away_xg = away_xg if away_xg > 0 else 0.1

        # Run the Monte Carlo / Poisson simulation
        top_scorelines, match_odds = simulate_match(home_xg, away_xg)
        
        return jsonify({
            'fixture_id': fixture_id,
            'live_metrics': {
                'home_xg': home_xg,
                'away_xg': away_xg
            },
            'match_odds': match_odds,
            'most_likely_scorelines': top_scorelines
        })
        
    except Exception as e:
        return jsonify({'error': f"Failed to run simulation: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)