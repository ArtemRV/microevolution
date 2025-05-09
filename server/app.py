import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify, send_file, render_template
from flask_socketio import SocketIO, emit
import os
from werkzeug.utils import secure_filename
import uuid
import torch
import numpy as np
from common.settings import default_settings
from common.core import Environment
from common.models import Actor
from common.utils import logging
from server.simulation import run_simulation

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Directories
UPLOAD_FOLDER = 'models'
SIMULATION_FOLDER = os.path.join('server', 'simulations')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIMULATION_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class DDPGAgent:
    def __init__(self, state_dim, action_dim, settings):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.settings = settings

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        return np.clip(action, -1, 1)

    def load_model(self, actor_path):
        try:
            self.actor.load_state_dict(torch.load(actor_path, weights_only=True))
            logging.info(f"Loaded actor model from {actor_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_model():
    try:
        if 'actor' not in request.files:
            return jsonify({'error': 'Actor model is required'}), 400

        actor_file = request.files['actor']
        if actor_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        actor_filename = secure_filename(actor_file.filename)
        actor_path = os.path.join(app.config['UPLOAD_FOLDER'], f'actor_{uuid.uuid4()}_{actor_filename}')
        actor_file.save(actor_path)

        env = Environment(default_settings)
        state_dim = len(env.reset())
        agent = DDPGAgent(state_dim, 2, default_settings)
        agent.load_model(actor_path)

        return jsonify({
            'message': 'Model uploaded successfully',
            'actor_path': actor_path
        })
    except Exception as e:
        logging.error(f"Error uploading model: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('start_simulation')
def handle_simulation(data):
    try:
        actor_path = data.get('actor_path')
        lives = int(data.get('lives', 3))
        sid = request.sid

        if not actor_path:
            emit('simulation_error', {'error': 'Actor path is required'})
            return

        env = Environment(default_settings)
        state_dim = len(env.reset())
        agent = DDPGAgent(state_dim, 2, default_settings)
        agent.load_model(actor_path)

        results, plot_path = run_simulation(agent, default_settings, lives, sid, socketio)

        emit('simulation_complete', {
            'results': results,
            'plot_url': f'/simulations/{os.path.basename(plot_path)}'
        }, room=sid)
    except Exception as e:
        logging.error(f"Error running simulation: {e}")
        emit('simulation_error', {'error': str(e)}, room=sid)

@app.route('/simulations/<filename>')
def serve_output(filename):
    return send_file(os.path.join('simulations', filename))

if __name__ == '__main__':
    socketio.run(app, debug=True)