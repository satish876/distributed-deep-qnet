from flask import Flask, Response, jsonify, request, render_template
import modman
from datetime import datetime, timedelta


# GLOBAL VARS
CENTRAL_MODEL = {}
LEARNING_RATE = 0.001
ITERATION = -1
ALL_PARAMS=[]
SCORES={}
U_TIME_STAMP = None
WTS=10

# Client class to manage updates
class CParamas:
    client_key = None
    iteration = None
    steps = None
    params= None
    mem_size = None

# LOCK VAR
MODEL_LOCK = False

#Start  Supporting functions
def register(add):
    SCORES[add]==1
def get_score(add):
    if add in SCORES:
        return SCORES[add]
    else:
        register(add)
        return 1
def correctness(client_params):
    if SCORES[client_params.client_key] >0 and client_params.params != None:
        return True, [client_params.params, client_params.mem_size]
    else:
        return False, []
def collect_params():
    global ALL_PARAMS
    global SCORES
    all_params=[]
    for x in ALL_PARAMS:
        valid, params = correctness(x)
        if valid:
            SCORES[x.client] +=1
            all_params.append(params)
        else:
            SCORES[x.client] -=1

#End Supporting functions

app = Flask(__name__)


@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/model/get', methods=['GET'])
def get_model():
    global CENTRAL_MODEL
    global ITERATION
    payload = {
        'params': CENTRAL_MODEL,
        'iteration': ITERATION
    }

    # Waiting for MODEL UPDATES
    global MODEL_LOCK
    while MODEL_LOCK:
        pass

    return jsonify(payload)


@app.route('/api/model/set', methods=['POST'])
def set_model():
    global CENTRAL_MODEL
    global ITERATION
    global LEARNING_RATE
    if ITERATION>=0:
        return jsonify({'iteration': ITERATION, 'Message': 'Error Model Already exist.'})
    params = request.get_json()

    print(
        f'Got Model Params from Client ID = {params["pid"]} IP Address = {request.remote_addr}')

    # Update ITERATION
    ITERATION += 1

    # Set CENTRAL MODEL params
    global MODEL_LOCK
    MODEL_LOCK = True
    CENTRAL_MODEL = params['model']
    LEARNING_RATE = params['learning_rate']
    MODEL_LOCK = False

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Model Params Set.'})

def update_model():
    global CENTRAL_MODEL
    global LEARNING_RATE
    global ITERATION
    global ALL_PARAMS
    list_of_params =   collect_params()
    # Update ITERATION
    ITERATION += 1

    # Apply Gradients and Update CENTRAL MODEL
    global MODEL_LOCK
    MODEL_LOCK = True
    CENTRAL_MODEL = modman.Federated_average(list_of_params)
    MODEL_LOCK = False
    ALL_PARAMS = []
    # PRINT RESPONSE
    print('iteration :', ITERATION,' Updated Model Params.')

@app.route('/api/model/post_params', methods=['POST'])
def post_params():
    update_params = request.get_json()
    key=request.remote_addr+":"+update_params["pid"]
    c_params = CParamas()
    c_params.client_key=key
    c_params.params=update_params['params']
    c_params.mem_size=update_params['mem_size']
    c_params.iteration=update_params['iteration']
    print(
        f'Got Parameters from Client ID = {update_params["pid"]} IP Address = {request.remote_addr}')

    global ALL_PARAMS
    global U_TIME_STAMP #updating time stamp
    global WTS #waiting time stamp
    if (len(ALL_PARAMS))==0:
        U_TIME_STAMP=datetime.now()+timedelta(seconds=WTS)
    elif U_TIME_STAMP<datetime.now() or len(ALL_PARAMS)==4 :
        update_model()
    # Storing params
    ALL_PARAMS.append(c_params)

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Collected Model Params.'})


if __name__ == "__main__":
    app.run(debug=True, port=5500)
