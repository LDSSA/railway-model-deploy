import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, DateTimeField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configuration
DEBUG_MODE = os.environ.get('DEBUG', '0') == '1'

########################################
# Feature Ranges for Validation
########################################
feature_ranges = {
    "sku": (1124.0, 4735.0),
    "time_key": (20230103.0, 20241028.0),
    "sales_volume": (0.0, 157854.0),
    "chain_price": (4.34, 780.89),
    "competitorA_campaign": (0.0, 0.0),
    "competitorB_campaign": (0.0, 0.0),
    "day_of_week": (0.0, 6.0),
    "is_weekend": (0.0, 1.0),
    "month": (1.0, 12.0),
    "quarter": (1.0, 4.0),
    "is_month_end": (0.0, 1.0),
    "is_month_start": (0.0, 1.0),
    "day_sin": (-0.9749279121818236, 0.9749279121818236),
    "day_cos": (-0.9009688679024191, 1.0),
    "month_sin": (-1.0, 1.0),
    "month_cos": (-1.0, 1.0),
    "pvp_final_competitorA_ma7": (4.34, 780.89),
    "pvp_final_competitorB_ma7": (3.99, 555.297948),
    "pvp_final_competitorA_ma14": (4.34, 780.89),
    "pvp_final_competitorB_ma14": (3.99, 555.297948),
    "pvp_final_competitorA_ma30": (4.34, 780.89),
    "pvp_final_competitorB_ma30": (3.9901045333333327, 555.297948),
    "price_diff_A_vs_B": (-104.49, 197.87938899999995),
    "price_ratio_A_vs_B": (0.3331844573470299, 3.156063786422382),
    "price_diff_A_vs_chain": (-303.720732, 212.62273200000004),
    "price_ratio_A_vs_chain": (0.3331844573470299, 4.125412541254127),
    #"pvp_is_competitorA": (4.34, 780.89),
    #"pvp_is_competitorB": (3.99, 555.297948),
}


def check_range(name, value):
    minv, maxv = feature_ranges[name]
    if not (minv <= value <= maxv):
        raise ValueError(f"{name} {value} out of range [{minv}, {maxv}]")

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
    
    # Type validation that accepts both strings and numbers
    try:
        sku = int(req['sku'])  # Convert to int (works for both "123" and 123)
        time_key = int(req['time_key'])
        # --- Range checks here ---
        check_range("sku", sku)
        check_range("time_key", time_key)
        if require_actual:
            pvpA = float(req['pvp_is_competitorA_actual'])
            pvpB = float(req['pvp_is_competitorB_actual'])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid field types or out of range: {str(e)}")
    
    # Additional validation
    if sku <= 0:
        raise ValueError("SKU must be a positive number")
    if time_key <= 0:
        raise ValueError("time_key must be a positive number")
    
    return True

def prepare_features(req):
    """Prepare features for model prediction"""
    features_dict = {
        "sku": float(int(req['sku'])),  # Convert to int first, then float
        "time_key": float(int(req['time_key']))
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
        
        sku = int(req['sku'])  # Convert to int (handles both string and number inputs)
        time_key = int(req['time_key'])
        
        # Check for existing prediction
        existing = PricePrediction.select().where(
            (PricePrediction.sku == sku) & 
            (PricePrediction.time_key == time_key)
        ).first()
        
        if existing:
            return jsonify({
                "error": "Prediction already exists",
                "existing_prediction": model_to_dict(existing, exclude=[
                    'id', 'prediction_time', 'actual_pvpA', 'actual_pvpB'
                ])
            }), 409
        
        # Make prediction
        features = prepare_features(req)
        y_pred = pipeline.predict(features)
        y_pred = y_pred.flatten() if hasattr(y_pred, "flatten") else y_pred
        
        if len(y_pred) != 2:
            raise ValueError("Model returned unexpected number of predictions")
        
        # Store prediction
        prediction = PricePrediction.create(
            sku=sku,
            time_key=time_key,
            predicted_pvpA=float(y_pred[0]),
            predicted_pvpB=float(y_pred[1])
        )
        
        return jsonify(model_to_dict(prediction, exclude=[
            'id', 'prediction_time', 'actual_pvpA', 'actual_pvpB'
        ])), 201
        
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

@app.route('/predictions/<sku>/<time_key>', methods=['GET'])
def get_prediction(sku, time_key):
    """Retrieve a specific prediction"""
    try:
        # Convert parameters to integers (handles both string and number inputs)
        sku_int = int(sku)
        time_key_int = int(time_key)
        
        prediction = PricePrediction.get(
            (PricePrediction.sku == sku_int) & 
            (PricePrediction.time_key == time_key_int)
        )
        return jsonify(model_to_dict(prediction)), 200
    except ValueError:
        return jsonify({'error': 'SKU and time_key must be numbers'}), 400
    except PricePrediction.DoesNotExist:
        return jsonify({
            'error': f'No prediction found for sku {sku} and time_key {time_key}'
        }), 404
    except Exception as e:
        app.logger.error(f"Retrieval error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

########################################
# Additional Endpoints
########################################

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/')
def api_info():
    return jsonify({
        'name': 'Price Prediction API',
        'version': '1.0',
        'endpoints': {
            'POST /forecast_prices': 'Make new predictions',
            'POST /actual_prices': 'Update with actual prices',
            'GET /predictions/<sku>/<time_key>': 'Get specific prediction'
        }
    })

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

