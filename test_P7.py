import sys
import os
import json

import pytest
import pickle
import pandas as pd
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import warnings

# Ajoutez le chemin d'accès au répertoire 'app' contenant md.py
#sys.path.append(os.path.abspath('..\\app'))

#print(os.getcwd())
  
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Opening JSON file
f = open('data.json')
  
# returns JSON object as a dictionary
data = json.load(f)

accepted = [data['accepted']]
refused = [data['refused']]

def test_predict_accepted():
  # Arrange
  loaded_model = mlflow.sklearn.load_model("model")
  print(loaded_model)

  # Act
  outcome = loaded_model.predict(accepted)
  # Assert
  assert outcome == 0

def test_predict_refused():
  # Arrange
  loaded_model = mlflow.sklearn.load_model("model")

  # Act
  outcome = loaded_model.predict(refused)
  # Assert
  assert outcome == 1
  '''

def test_verif():

  # Act
  outcome = 0
  # Assert
  assert outcome == 0'''





  


