from flask import Flask, request, jsonify, render_template
from src.utils import predict_burnout, load_utils

app = Flask(__name__)
__d = {1: "Medium", 2: "High", 0: "Low"}

@app.post('/predict')
def callPredict():
    data = request.get_json()
    res = predict_burnout(data)[0]

    return jsonify(
        {"message": f"Your burnout level is {__d[res]}"}
    )

@app.get('/')
def homepage():
    return render_template('index.html')


if __name__ == '__main__':
    load_utils()
    app.run()