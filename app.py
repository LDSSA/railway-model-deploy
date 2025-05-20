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

