from flask import Flask, Response, jsonify, request, render_template


# GLOBAL VARS
CENTRAL_MODEL = {}
ITERATION = 0


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
    return jsonify(payload)


@app.route('/api/model/update', methods=['POST'])
def update_model():
    updateParams = request.json

    global CENTRAL_MODEL
    global ITERATION
    ITERATION += 1

    CENTRAL_MODEL = updateParams

    return Response("Updated", status=200)


if __name__ == "__main__":
    app.run(debug=True, port=5500)
