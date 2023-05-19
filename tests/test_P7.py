import sys
import os

# Ajoutez le chemin d'accès au répertoire 'app' contenant md.py
sys.path.append(os.path.abspath('..\\app'))

# Importez le module 
#import api

import pytest
import pickle
import pandas as pd
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import warnings
  
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

#df = pd.read_csv('../data/df.csv')

accepted = [[ 8.06802486e-02, -2.37000000e+03, -5.70000000e+01,  0.00000000e+00,
        0.00000000e+00,  3.62569061e-01, -2.39667570e+03,  1.00000000e+00,
        2.60640000e+05, -2.08410000e+04]]

refused = [[ 6.77215190e-02, -7.80000000e+01, -2.03000000e+02,
         0.00000000e+00,  0.00000000e+00,  1.26582278e+00,
        -4.89000000e+02,  0.00000000e+00,  1.42200000e+05,
        -1.17050000e+04]]

def test_predict_accepted():
  # Arrange
  loaded_model = mlflow.sklearn.load_model("../model")

  # Act
  outcome = loaded_model.predict(accepted)
  # Assert
  assert outcome == 0

def test_predict_refused():
  # Arrange
  loaded_model = mlflow.sklearn.load_model("../model")

  # Act
  outcome = loaded_model.predict(refused)
  # Assert
  assert outcome == 1
  '''

def test_predict_r():
  # Arrange
  #loaded_model = mlflow.sklearn.load_model("../model")

  #refused_ = pd.DataFrame(refused,index=[0])
  # Act
  outcome = api.predict(10002)
  # Assert
  assert outcome == 0'''


