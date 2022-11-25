from app import app
from flask import jsonify
from flask import flash, request
import pandas as pd

from src.functions.api.add_new_models import add_new_models


@app.route('/add_new_models/', methods=['GET', 'POST'])
def run_add_new_models():
    new_models_winners = request.json
    response = add_new_models(new_models_winners)
    return response


@app.errorhandler(404)
def showMessage(error=None):
    message = {
        'status': 404,
        'message': 'Record not found: ' + request.url,
    }
    respone = jsonify(message)
    respone.status_code = 404
    return respone


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
