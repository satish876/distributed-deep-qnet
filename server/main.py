from flask import Flask, Response, jsonify, request, render_template
import modman


# GLOBAL VARS
CENTRAL_MODEL = {}
LEARNING_RATE = 0
ITERATION = -1

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
        'params': CENTRAL_MODEL,
        'iteration': ITERATION
    }

    # Waiting for MODEL UPDATES
    global MODEL_LOCK
    while MODEL_LOCK:
        print("WAITING FOR MODEL UPDATE!!!")

    return jsonify(payload)


@app.route('/api/model/set', methods=['POST'])
def update_model():
    params = request.json

    global CENTRAL_MODEL
    global ITERATION
    global LEARNING_RATE

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


@app.route('/api/model/update', methods=['POST'])
def update_model():
    update_params = request.json

    global CENTRAL_MODEL
    global LEARNING_RATE
    global ITERATION

    # Update ITERATION
    ITERATION += 1

    # Apply Gradients and Update CENTRAL MODEL
    global MODEL_LOCK
    MODEL_LOCK = True
    CENTRAL_MODEL = modman.update_model(
        update_params, CENTRAL_MODEL, LEARNING_RATE)
    MODEL_LOCK = False

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Updated Model Params.'})


if __name__ == "__main__":
    app.run(debug=True, port=5500)
