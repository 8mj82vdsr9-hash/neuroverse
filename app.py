from flask import Flask, jsonify, request, send_from_directory
from neuroverse.neural_core import FeedForwardNetwork
from neuroverse.environment import SimpleEnvironment
from neuroverse.memory import MemorySystem
from neuroverse.experiment_manager import ExperimentManager

app = Flask(__name__, static_folder='static')

network = FeedForwardNetwork([2, 5, 1])
env = SimpleEnvironment(width=6, height=6)
memory = MemorySystem(max_entries=300)
experiment_manager = ExperimentManager(network=network, memory=memory, env=env)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/network')
def api_network():
    return jsonify(network.state())

@app.route('/api/env')
def api_env():
    return jsonify(env.state())

@app.route('/api/memory')
def api_memory():
    return jsonify(memory.state())

@app.route('/api/step', methods=['POST'])
def api_step():
    payload = request.get_json(silent=True) or {}
    inputs = payload.get('input', [0.0, 0.0])
    output = network.forward(inputs)
    return jsonify({'input': inputs, 'output': output})

@app.route('/api/train_xor', methods=['POST'])
def api_train_xor():
    payload = request.get_json(silent=True) or {}
    epochs = int(payload.get('epochs', 3000))
    lr = float(payload.get('lr', 0.25))
    history = network.train_xor(epochs=epochs, lr=lr)
    return jsonify({
        'epochs': epochs,
        'lr': lr,
        'final_loss': history[-1] if history else None,
        'history': history,
    })

@app.route('/api/env/reset')
def env_reset():
    env.reset()
    return jsonify(env.state())

@app.route('/api/env/step', methods=['POST'])
def env_step():
    payload = request.get_json(silent=True) or {}
    action = payload.get('action', 'stay')
    observation, reward, done, info = env.step(action)
    memory.add_experience({
        'observation': observation,
        'action': action,
        'reward': reward,
        'done': done,
        'info': info,
    })
    return jsonify({
        'observation': observation,
        'reward': reward,
        'done': done,
        'info': info,
        'state': env.state(),
    })

@app.route('/api/experiment', methods=['POST'])
def experiment_route():
    payload = request.get_json(silent=True) or {}
    mode = payload.get('mode', 'xor')
    epochs = int(payload.get('epochs', 3000))
    result = experiment_manager.run(mode=mode, epochs=epochs)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
