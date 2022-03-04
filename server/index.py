from logging import exception
from flask import Flask, Response, jsonify, request, render_template

import modman
from datetime import datetime, timedelta

import numpy as np

app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

ITERATION = -1
QUEUE = []
UPDATES = {}
LOCK = False
KEYS = ['SEQL.0.weight', 'SEQL.0.bias', 'SEQL.2.weight', 'SEQL.2.bias', 'SEQL.4.weight', 'SEQL.4.bias', 'SEQL.6.weight', 'SEQL.6.bias']
GLOBAL_PARAMS = {_:None for _ in KEYS}

@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/model/get', methods=['GET'])
def get_model():
    global GLOBAL_PARAMS, LOCK
    while LOCK:
        pass
    payload = {
        'params': GLOBAL_PARAMS,
        'iteration': ITERATION
    }
    return jsonify(payload)


@app.route('/api/model/set', methods=['POST'])
def set_model():
    global GLOBAL_PARAMS, ITERATION, KEYS, LOCK
    while LOCK:
        pass
    params = request.get_json()

    GLOBAL_PARAMS = params['model']#{key: params['model'][key].tolist() for key in KEYS}
    ITERATION = 0

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Model Params Set.'})


@app.route('/api/model/post_params', methods=['POST'])
def post_params():
    global GLOBAL_PARAMS, ITERATION, UPDATES, LOCK
    while LOCK:
        pass
    params = request.get_json()
    client_iteration = params['iteration']

    '''
        1. accept the parameter from the client and release it
    '''

    if client_iteration > ITERATION:
        # client has sent the future updates, handle this
        return jsonify({'iteration': ITERATION, 'Message': 'Rejected'})
    elif client_iteration < ITERATION:
        # this is outdated updates
        # pass
        return jsonify({'iteration': ITERATION, 'Message': 'Rejected'})
    else:
        client_id = get_client_id(params)
        if client_id in UPDATES and UPDATES[client_id][2] >= client_iteration:
            return jsonify({'iteration': ITERATION, 'Message': 'Rejected'})
            print('='*30, "BUG", "="*30, client_id, client_iteration, ITERATION)
        
        UPDATES[client_id] = [params['model'], params['mem_size'], client_iteration]
        if len(UPDATES) == 3:
            federated_average()
            print("client", "*"*30, ">", client_id, client_iteration, ITERATION)
            UPDATES = {}
        return jsonify({'iteration': ITERATION, 'Message': 'Collected Model Params.'})


def get_client_id(params):
    return request.remote_addr + ":" + str(params["pid"])

def federated_average():
    global UPDATES, GLOBAL_PARAMS, LOCK, ITERATION, KEYS
    LOCK = True
    result = {}
    total_count = 0

    for _ in UPDATES:
        total_count += UPDATES[_][1]

    for key in KEYS:
        # discount_factor = 
        try:
            arr = [ np.multiply(UPDATES[_][1]/total_count, UPDATES[_][0][key]) for _ in UPDATES]
            result[key] = np.sum(arr, axis=0).tolist()
        except Exception as e:
            print("="*100)
            print("="*100)
            print(e)
            print("\n","="*100)
            print("\n","="*100)
    
    GLOBAL_PARAMS = result
    ITERATION += 1
    LOCK = False


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5501)
