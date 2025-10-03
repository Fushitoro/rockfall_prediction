from flask import Flask, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

history = []

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/api/latest')
def get_latest():
    try:
        response = requests.get('http://localhost:5000/simulate-and-predict', timeout=5)
        data = response.json()
        history.append(data)
        if len(history) > 50:
            history.pop(0)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': 'Failed to get latest data', 'details': str(e)}), 503

@app.route('/api/history')
def get_history():
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True, port=8081)
