from flask import Flask, Response, jsonify, request, render_template
from ib_insync import client
import modman
from datetime import datetime, timedelta

import numpy as np

app = Flask(__name__)

GLOBAL_PARAMS = {}
ITERATION = -1
QUEUE = []
UPDATES = []
LOCK = False

@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/model/get', methods=['GET'])
def get_model():
    while LOCK: pass
    global GLOBAL_PARAMS
    payload = {
        'params': GLOBAL_PARAMS,
        'iteration': ITERATION
    }
    return jsonify(payload)


@app.route('/api/model/set', methods=['POST'])
def set_model():
    while LOCK: pass
    global GLOBAL_PARAMS, ITERATION
    params = request.get_json()

    GLOBAL_PARAMS = params['model']
    ITERATION += 1

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Model Params Set.'})


@app.route('/api/model/post_params', methods=['POST'])
def post_params():
    while LOCK: pass
    global GLOBAL_PARAMS, ITERATION, UPDATES
    params = request.get_json()
    client_iteration = params['ITERATION']

    '''
        1. accept the parameter from the client and release it
    '''
    
    if client_iteration > ITERATION:
        # client has sent the future updates, handle this
        return jsonify({'iteration': ITERATION , 'Message': 'Rejected'})
    elif client_iteration < ITERATION:
        # this is outdated updates
        # pass
        return jsonify({'iteration': ITERATION , 'Message': 'Rejected'})
    else:
        UPDATES.append(params['model'])
        if len(UPDATES) == 2:
            federated_average()
            UPDATES = []
        return jsonify({'iteration': ITERATION , 'Message': 'Collected Model Params.'})

def federated_average():
    global UPDATES, GLOBAL_PARAMS, LOCK, ITERATION
    LOCK = True
    keys = UPDATES[0].keys()
    result = {}
    total_count = 0
    
    for u in UPDATES:
        total_count += u[1]
    
    for key in keys:
        result[key] = np.sum([ (_[1]/total_count)*_[0] for _ in UPDATES])
    GLOBAL_PARAMS = result
    ITERATION += 1
    LOCK = False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5501)
