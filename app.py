# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:12:59 2020

@author: Asus
"""

# the Flask object is for creating an HTTP server
# the request object holds all of the contents of an HTTP request that 
# someone is making
# the jsonify function is useful for when we want to return json from the
# function we are using.
from flask import Flask, request, jsonify
import joblib
import json
import pickle
import pandas as pd
import os
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

##################################
# sets up database
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError,
)

#DB = SqliteDatabase('predictions.db')
# the connect function checks if there is a DATABASE_URL env var
# if it exists, it uses it to connect to a remote postgres db
# otherwise, it connects to a local sqlite db stored in the predictions.db file
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    obs_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null='null')

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

##################################
# unpickles the previously trained model
with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')

##################################
# sets up webserver

# the Flask constructor creates a new application that we can add routes to
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    _id = obs_dict['id']
    observation = obs_dict['observation']
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]

    response = {'proba': proba}
    
    p = Prediction(
        obs_id = _id,
        proba = proba,
        observation = request.data,
        )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
            
    return jsonify(
        response
    )

@app.route('/update', methods=['POST'])
def update():
    obs_dict = request.get_json()
    try:
        p = Prediction.get(Prediction.obs_id == obs_dict['id'])
        p.true_class = obs_dict['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs_dict['id'])
        return jsonify({'error': error_msg})

########################################

if __name__ == "__main__":
    app.run(debug=True)