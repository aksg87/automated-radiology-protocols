import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

data = pd.read_csv("./100lines.csv")

le = preprocessing.LabelEncoder()

le.fit(data['Protocol'])

list(le.classes_)

encoded_P = le.transform(data['Protocol'])

onehot_P = np_utils.to_categorical(encoded_P)