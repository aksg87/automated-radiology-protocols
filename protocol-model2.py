import numpy as np
import pandas as pd

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

data = pd.read_csv("./65k lines.csv")

docs = data['Diagnosis']
labels = data['Protocol']

le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels = np_utils.to_categorical(labels)
num_labels = len(le.classes_)

vocab_size = 200

encoded_docs = [one_hot(d, vocab_size) for d in docs]

print(encoded_docs)

max_length = 25

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_docs)

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(num_labels, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=2)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=2)
print('Accuracy: %f' % (accuracy*100))

