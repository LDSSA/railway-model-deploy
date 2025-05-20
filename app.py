import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


import json
import joblib
import pickle
import pandas as pd
from flask import Flask, request, jsonify


import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, DateTimeField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configuration
DEBUG_MODE = os.environ.get('DEBUG', '0') == '1'

########################################
# Database Setup
########################################

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class PricePrediction(Model):
    sku = IntegerField()
    time_key = IntegerField()
    predicted_pvpA = FloatField()
    predicted_pvpB = FloatField()
    actual_pvpA = FloatField(null=True)
    actual_pvpB = FloatField(null=True)
    prediction_time = DateTimeField(default=datetime.now)
    
    class Meta:
        database = DB
        indexes = (
            (('sku', 'time_key'), True),  # composite unique index
        )

DB.create_tables([PricePrediction], safe=True)

########################################
# Model Loading
########################################

# Load model artifacts
try:
    with open('columns.json') as fh:
        columns = json.load(fh)
    
    pipeline = joblib.load('pipeline.pickle')
    
    with open('dtypes.pickle', 'rb') as fh:
        dtypes = pickle.load(fh)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

########################################
# Helper Functions
########################################

def validate_price_request(req, require_actual=False):
    """Validate price prediction/actual data request"""
    if not isinstance(req, dict):
        raise ValueError("Request must be a JSON object")
    
    required_fields = {'sku', 'time_key'}
    if require_actual:
        required_fields.update({'pvp_is_competitorA_actual', 'pvp_is_competitorB_actual'})
    
    missing_fields = required_fields - set(req.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Type validation
    try:
        sku = int(req['sku'])
        time_key = int(req['time_key'])
        if require_actual:
            pvpA = float(req['pvp_is_competitorA_actual'])
            pvpB = float(req['pvp_is_competitorB_actual'])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid field types: {str(e)}")
    
    return True

def prepare_features(req):
    """Prepare features for model prediction"""
    features_dict = {
        "sku": float(req['sku']),
        "time_key": float(req['time_key'])
    }
    
    # Add default values for any missing columns
    for col in columns:
        if col not in features_dict:
            features_dict[col] = 0.0
    
    return pd.DataFrame([features_dict], columns=columns).astype(dtypes)

########################################
# API Endpoints
########################################

@app.route('/forecast_prices/', methods=['POST'])
def forecast_prices():
    """Make price predictions and store results"""
    try:
        req = request.get_json()
        validate_price_request(req)
        
        sku = int(req['sku'])
        time_key = int(req['time_key'])
        
        # Check for existing prediction
        existing = PricePrediction.select().where(
            (PricePrediction.sku == sku) & 
            (PricePrediction.time_key == time_key)
        ).first()
        
        if existing:
            return jsonify({
                "error": "Prediction already exists",
                "existing_prediction": model_to_dict(existing)
            }), 409
        
        # Make prediction
        features = prepare_features(req)
        y_pred = pipeline.predict(features)
        y_pred = y_pred.flatten() if hasattr(y_pred, "flatten") else y_pred
        
        # Store prediction
        prediction = PricePrediction.create(
            sku=sku,
            time_key=time_key,
            predicted_pvpA=float(y_pred[0]),
            predicted_pvpB=float(y_pred[1])
        )
        
        return jsonify(model_to_dict(prediction)), 201
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/actual_prices/', methods=['POST'])
def actual_prices():
    """Update with actual price data"""
    try:
        req = request.get_json()
        validate_price_request(req, require_actual=True)
        
        sku = int(req['sku'])
        time_key = int(req['time_key'])
        
        # Find existing prediction to update
        try:
            prediction = PricePrediction.get(
                (PricePrediction.sku == sku) & 
                (PricePrediction.time_key == time_key)
            )
        except PricePrediction.DoesNotExist:
            return jsonify({
                'error': f'No prediction found for sku {sku} and time_key {time_key}'
            }), 404
        
        # Update with actual prices
        prediction.actual_pvpA = float(req['pvp_is_competitorA_actual'])
        prediction.actual_pvpB = float(req['pvp_is_competitorB_actual'])
        prediction.save()
        
        return jsonify(model_to_dict(prediction)), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Update error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predictions/<int:sku>/<int:time_key>', methods=['GET'])
def get_prediction(sku, time_key):
    """Retrieve a specific prediction"""
    try:
        prediction = PricePrediction.get(
            (PricePrediction.sku == sku) & 
            (PricePrediction.time_key == time_key)
        )
        return jsonify(model_to_dict(prediction)), 200
    except PricePrediction.DoesNotExist:
        return jsonify({
            'error': f'No prediction found for sku {sku} and time_key {time_key}'
        }), 404
    except Exception as e:
        app.logger.error(f"Retrieval error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

########################################
# Error Handlers
########################################

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Not Found',
        'message': str(e)
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500

########################################
# Application Startup
########################################

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=DEBUG_MODE
    )




""" The remaining code is inactivated
########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)

'''
@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    _id = obs_dict['id']
    observation = obs_dict['observation']
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


'''
@app.route('/predict', methods=['POST'])
def predict():
    # Deserialize the JSON payload
    obs_dict = request.get_json()

    # Validate the presence of required fields
    if not obs_dict or 'id' not in obs_dict or 'observation' not in obs_dict:
        return jsonify({'error': 'Payload must contain "id" and "observation" fields'}), 400

    _id = obs_dict['id']
    observation = obs_dict['observation']

    # Validate the observation fields
    required_fields = ['age', 'education', 'hours-per-week', 'native-country']
    for field in required_fields:
        if field not in observation:
            return jsonify({'error': f'Observation is missing required field: {field}'}), 400

    # Validate data types and formats
    try:
        age = int(observation['age'])
        if age < 0 or age > 120:
            return jsonify({'error': 'Age must be a positive integer between 0 and 120'}), 400

        education = str(observation['education'])
        hours_per_week = int(observation['hours-per-week'])
        if hours_per_week < 0 or hours_per_week > 168:
            return jsonify({'error': 'Hours-per-week must be a positive integer between 0 and 168'}), 400

        native_country = str(observation['native-country'])
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid data type for one or more fields: {str(e)}'}), 400

    # Create a DataFrame for the observation
    obs = pd.DataFrame([{
        'age': age,
        'education': education,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }], columns=columns).astype(dtypes)

    # Get the predicted probability
    proba = pipeline.predict_proba(obs)[0, 1]

    # Check if the observation ID already exists in the database
    if Prediction.select().where(Prediction.observation_id == _id).exists():
        existing_prediction = Prediction.get(Prediction.observation_id == _id)
        return jsonify({
            'error': f'Observation ID: "{_id}" already exists',
            'proba': existing_prediction.proba
        }), 409

    # Save the new prediction to the database
    p = Prediction(
        observation_id=_id,
        observation=json.dumps(observation),  # Store observation as a JSON string
        proba=proba,
        true_class=None  # true_class is null for now
    )
    try:
        p.save()
    except IntegrityError:
        return jsonify({'error': f'Observation ID: "{_id}" already exists'}), 409

    # Return the predicted probability
    return jsonify({'proba': proba}), 200



@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()

    # Validate the request payload
    if not obs or 'id' not in obs or 'true_class' not in obs:
        return jsonify({'error': 'Invalid request payload. Missing id or true_class.'}), 400

    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p)), 200  # Return 200 OK

    except Prediction.DoesNotExist:
        error_msg = f'Observation ID {obs['id']} does not exist'
        return jsonify({'error': error_msg}), 404 #Return 404 Not Found

    except Exception as e:
        # Handle other potential errors (database errors, etc.)
        error_msg = f'An unexpected error occurred: {str(e)}'
        return jsonify({'error': error_msg}), 500  # Return 500 Internal Server Error

'''
@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = f'Observation ID {obs['id']} does not exist'
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])
'''


# End webserver stuff
########################################


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)

""""
