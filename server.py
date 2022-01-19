from flask import Flask, Response, jsonify, request, render_template
from datetime import datetime, timedelta
# from pytest import param

import torch
app = Flask(__name__)

R = [0]*8
MODEL = [torch.empty(128, 4).normal_(mean=0, std=1), torch.ones(128),
         torch.empty(128, 128).normal_(mean=0, std=1), torch.ones(128),
         torch.empty(128, 128).normal_(mean=0, std=1), torch.ones(128),
         torch.empty(2, 128).normal_(mean=0, std=1), torch.ones(2)]
ITERATION = 0
LOCK = False
KEYS = ['SEQL.0.weight', 'SEQL.0.bias', 'SEQL.2.weight', 'SEQL.2.bias', 'SEQL.4.weight', 'SEQL.4.bias', 'SEQL.6.weight', 'SEQL.6.bias']


@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/model/set', methods=['POST'])
def set_model():
    global MODEL, LOCK, ITERATION, R, KEYS
    params = request.get_json()

    while LOCK:
        pass
    LOCK = True

    # KEYS = list(params['model'])
    # MODEL = get_theta(params['model'])
    R = [0]*len(MODEL)

    LOCK = False

    return jsonify({'iteration': ITERATION, 'Message': 'Model Params Set.'})


@app.route('/api/model/get', methods=['GET'])
def get_model():
    global LOCK

    while LOCK:
        pass
    LOCK = True
    response = get_response()
    LOCK = False

    return response


@app.route('/api/model/post_params', methods=['POST'])
def post_params():
    global LOCK, ITERATION
    params = request.get_json()
    if MODEL is None:
        while LOCK:
            pass
        LOCK = True
        response = get_response()
        LOCK = False
        return response

    RMSPROP(params['model'])
    ITERATION += 1

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Collected Model Params.'})


def get_theta(model):
    keys = list(model)
    M = []
    for k in keys:
        M.append(torch.tensor(model[k]))
    return M


def get_response():
    global MODEL, ITERATION

    M = []
    resp = {}
    if MODEL is not None:
        for i in range(len(MODEL)):
            resp[KEYS[i]] = MODEL[i].tolist()

    payload = {
        'params': resp,
        'iteration': ITERATION
    }

    return jsonify(payload)


def RMSPROP(params):
    global R, MODEL, LOCK
    if MODEL is None:
        return

    while LOCK:
        pass

    LOCK = True
    theta = get_theta(params)
    hadamard = [0.1*x*x for x in theta]
    R_discounted = [0.9*x for x in R]
    R = [R_discounted[i] + hadamard[i] for i in range(len(R_discounted))]
    MODEL = [MODEL[i] - 0.99*(theta[i]/torch.sqrt(R[i]))
             for i in range(len(R_discounted))]

    LOCK = False


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5501)
