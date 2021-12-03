from flask import Flask, Response, jsonify, request, render_template
import modman
import numpy as np
import torch

# GLOBAL MODEL VARS
CENTRAL_MODEL = {}
LEARNING_RATE = 0
ITERATION = -1
SQUARE_AVGS = {}

# LOCK VAR
MODEL_LOCK = False


app = Flask(__name__)


@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/model/get', methods=['GET'])
def get_model():
    global CENTRAL_MODEL
    global ITERATION
    payload = {
        'params': modman.convert_tensor_to_list(CENTRAL_MODEL),
        'iteration': ITERATION
    }

    # Waiting for MODEL UPDATES
    global MODEL_LOCK
    while MODEL_LOCK:
        pass

    return jsonify(payload)


@app.route('/api/model/set', methods=['POST'])
def set_model():
    params = request.get_json()

    print(
        f'Got Model Params from Client ID = {params["pid"]} IP Address = {request.remote_addr}')

    global CENTRAL_MODEL
    global ITERATION
    global LEARNING_RATE

    # Update ITERATION
    ITERATION += 1

    # Set CENTRAL MODEL params
    global MODEL_LOCK
    MODEL_LOCK = True
    set_model = params['model']
    LEARNING_RATE = params['learning_rate']
    if ITERATION <= 0:
        for key, value in set_model.items():
            CENTRAL_MODEL[key] = torch.Tensor(value)
            SQUARE_AVGS[key] = torch.zeros_like(
                CENTRAL_MODEL[key], memory_format=torch.preserve_format)
    MODEL_LOCK = False

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Model Params Set.'})


@app.route('/api/model/update', methods=['POST'])
def update_model():
    update_params = request.get_json()

    print(
        f'Got Gradients from Client ID = {update_params["pid"]} IP Address = {request.remote_addr}')

    global CENTRAL_MODEL
    global LEARNING_RATE
    global ITERATION
    global SQUARE_AVGS

    # Update ITERATION
    ITERATION += 1
    params = []
    grads = []
    square_avgs = []
    alpha = 0.99
    weight_decay = 0
    eps = 1e-8
    lr = LEARNING_RATE
    for key, value in CENTRAL_MODEL.items():
        params.append(value)
        square_avgs.append(SQUARE_AVGS[key])

    # Apply Gradients and Update CENTRAL MODEL
    grads = update_params['grads']
    global MODEL_LOCK
    MODEL_LOCK = True
    update_model = modman.RMSprop_update(params,
                                         grads,
                                         square_avgs,
                                         weight_decay,
                                         lr,
                                         eps,
                                         alpha)
    MODEL_LOCK = False

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Updated Model Params.'})


if __name__ == "__main__":
    app.run(debug=True, port=5501)
