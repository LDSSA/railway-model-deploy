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






# Predict endpoint
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
        DB.rollback()
        return jsonify({'error': f'Observation ID: "{_id}" already exists'}), 409

    # Return the predicted probability
    return jsonify({'proba': proba}), 200






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


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
