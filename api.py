from app import app
from flask import jsonify
from flask import flash, request
import pandas as pd

from src.functions.api.add_new_models import add_new_models


@app.route('/add_new_models/', methods=['GET', 'POST'])
def run_add_new_models():
    """
    Endpoint to add new models.

    This function is a Flask route that handles the '/add_new_models/' endpoint. It accepts GET and POST requests.

    Steps:
    1. Retrieve the JSON data from the request.
    2. Call the `add_new_models` function with the new models winners data.
    3. Return the response from the `add_new_models` function.

    Parameters:
    None

    Returns:
    The response from the `add_new_models` function.
    """
    new_models_winners = request.json
    response = add_new_models(new_models_winners)
    return response


@app.errorhandler(404)
def showMessage(error=None):
    """
    Handle 404 errors and return a JSON response.

    This function is a Flask error handler that is called when a 404 error occurs.

    Steps:
    1. Create a dictionary with the error status and message.
    2. Convert the dictionary to a JSON response using `jsonify`.
    3. Set the response status code to 404.
    4. Return the JSON response.

    Parameters:
    error (any): The error object (not used in this implementation).

    Returns:
    A JSON response with the error status and message.
    """
    message = {
        'status': 404,
        'message': 'Record not found: ' + request.url,
    }
    respone = jsonify(message)
    respone.status_code = 404
    return respone


if __name__ == "__main__":
    """
    Run the Flask application.

    This block of code is executed when the script is run directly (not imported as a module).

    Steps:
    1. Run the Flask application with the host set to "0.0.0.0" and the port set to 5000.

    Parameters:
    None

    Returns:
    None
    """
    app.run(host="0.0.0.0", port=5000)
