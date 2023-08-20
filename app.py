import json
from utils import predict
from flask import Flask, jsonify, request


app = Flask(__name__)
@app.post('/')
def index():
    data = request.data
    data = json.loads(data)
    text = data['text']
    score = predict(text)
    return jsonify({'pred': f'{score}'})

if __name__ == '__main__':
    app.run()