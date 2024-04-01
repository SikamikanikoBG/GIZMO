from flask import Flask
from flask_cors import CORS, cross_origin

"""
This module sets up a Flask application and enables CORS (Cross-Origin Resource Sharing) for the application.

Attributes:
- app (Flask): The Flask application instance.
- CORS (flask_cors.CORS): The CORS object that enables cross-origin resource sharing for the Flask application.

"""

app = Flask(__name__)
"""
Initialize a Flask application instance.

The Flask application instance is assigned to the `app` variable, which can be used to define routes, handle requests, and configure the application.

"""

CORS(app)
"""
Enable CORS (Cross-Origin Resource Sharing) for the Flask application.

The `CORS` object is initialized with the `app` instance, allowing the application to handle cross-origin requests.

"""
