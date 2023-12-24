from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load and preprocess data
data = np.loadtxt('datasets/tictac_multi.txt')
X = data[:, :9]
y = data[:, 9:]
scaler = StandardScaler().fit(X)

# Initialize and train MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=10000)
mlp.fit(scaler.transform(X), y)

def ML_move(board):
    preds = mlp.predict(scaler.transform([board]))
    move = np.argmax(preds)
    while board[move] != 0:
        preds[0][move] = -np.inf
        move = np.argmax(preds)
    return move

def check_winner(board):
    wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for line in wins:
        if board[line[0]] == board[line[1]] == board[line[2]] and board[line[0]] != 0:
            return True, line
    return False, None

def numpy_to_list(board):
    return [int(num) for num in board]

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Exception occurred: {e}", exc_info=True)
    return jsonify({'error': 'An error occurred'}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    try:
        data = request.json
        app.logger.debug(f"Received data: {data}")
        board = np.array(data['board'], dtype=int)
        player_move = data['move']
        
        if board[player_move] == 0:
            board[player_move] = 1
            winner, winning_line = check_winner(board)
            if winner:
                return jsonify({'status': 'win', 'board': numpy_to_list(board), 'winningLine': winning_line})
            elif 0 not in board:
                return jsonify({'status': 'draw', 'board': numpy_to_list(board)})
            else:
                ai_move = ML_move(board)
                board[ai_move] = -1
                winner, winning_line = check_winner(board)
                if winner:
                    return jsonify({'status': 'loss', 'board': numpy_to_list(board), 'winningLine': winning_line})
                else:
                    return jsonify({'status': 'continue', 'board': numpy_to_list(board)})
        else:
            return jsonify({'status': 'invalid', 'board': numpy_to_list(board)})
    except Exception as e:
        app.logger.error(f"Error in make_move: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
