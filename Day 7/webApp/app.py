from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get('/health_check')
def healthCheck():
    return {"healthcheck": "Ok"}

if __name__ == '__main__':
    app.run()