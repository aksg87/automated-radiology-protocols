import numpy as np
import pandas as pd

from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Input, Dense, Flatten, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import concatenate
from keras.utils import plot_model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

#input text
data = pd.read_csv("./65k.csv")

#define dx document text
docs = data['Diagnosis']

#define anatomy document text
anatomy = data['Anatomy'].astype(str)
le_anatomy = preprocessing.LabelEncoder()
le_anatomy.fit(anatomy)
anatomy = le_anatomy.transform(anatomy)
anatomy = np_utils.to_categorical(anatomy)
num_anatomytypes = len(le_anatomy.classes_)


#define exam type ordered
exam = data['Exam'].astype(str)
le_exam = preprocessing.LabelEncoder()
le_exam.fit(exam)
exam = le_exam.transform(exam)
exam = np_utils.to_categorical(exam)
num_examtypes = len(le_exam.classes_)


#define protocol labels
labels = data['Protocol']
original_labels = labels
le_proto = preprocessing.LabelEncoder()
le_proto.fit(labels)
labels = le_proto.transform(labels)
labels = np_utils.to_categorical(labels)
num_protolabels = len(le_proto.classes_)

def convert_coded_label(label):
	label = le_proto.inverse_transform(np.argmax(label))
	return label

#prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
print('vocab size', vocab_size)

#integer encode documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)

# pad documents to length of 25 words
max_length = 15
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# load the whole embedding into memory
embeddings_index = dict()
f = open('vectors.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

dim_len = len(coefs)
print('Dimension of vector %s.' % dim_len)

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, dim_len))
for word, i in tqdm(t.word_index.items()):
	embedding_vector = embeddings_index.get(word)

	if embedding_vector is not None and np.shape(embedding_vector) != (202,):
		embedding_matrix[i] = embedding_vector		
	if np.shape(embedding_vector) == (202,):
		print(i)
		print("embedding_vector", np.shape(embedding_vector))
		print("embedding_matrix", np.shape(embedding_matrix[i]))


# define keras model
diagnosis_input = Input(shape=(max_length,), dtype = 'int32', name='diagnosis_input')
x = Embedding(vocab_size, dim_len, weights=[embedding_matrix], input_length=max_length, trainable=False)(diagnosis_input)
x = Dropout(0.2)(x)
x = LSTM(30)(x)
x = Dropout(0.2)(x)

dense_out = Dense(64, activation='relu')(x)
anatomy_input = Input(shape=(13,), name='anatomy_input')
exam_input = Input(shape=(275,), name='exam_input')

x = concatenate([dense_out,anatomy_input,exam_input])

x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
main_output = Dense(num_protolabels, activation='softmax', name='main_output')(x)

model = Model(inputs=[diagnosis_input, anatomy_input, exam_input], outputs=[main_output])

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# summarize the model
print(model.summary())

# fit the model
model.fit([padded_docs,anatomy,exam], labels, epochs=50, verbose=2)
model.save('proto_model.h5')
# evaluate the model

loss, accuracy = model.evaluate(padded_docs, labels, verbose=2)
print('Accuracy: %f' % (accuracy*100))

def customAccuracy(model, padded_docs, anatomy, exam, labels):

	if np.ndim(labels[0])>0:
		labels = list(map(convert_coded_label, labels))

	preds = model.predict([padded_docs,anatomy,exam])
	converted_preds = list(map(convertSoftmax, preds))
	untuple = lambda x_y: x_y[0]
	converted_preds = list(map(untuple, converted_preds))

	results_compare = list(zip(converted_preds, labels))
	checkProtos = lambda protos_element: protos_element[0].__contains__(protos_element[1])
	results = list(map(checkProtos, results_compare))

	return (sum(results) / len(results)), results

def queryModel(model, txt):
	converted_txt = convertQuery(txt)
	pred = model.predict(converted_txt)
	answer = convertSoftmax(pred)
	return answer
	
def convertQuery(txt):
	temp_docs = []
	temp_docs.append(txt)
	#integer encode documents

	temp_encoded_docs = t.texts_to_sequences(temp_docs)
	print(temp_encoded_docs)

	# pad documents to length of 25 words
	max_length = 15
	temp_padded_docs = pad_sequences(temp_encoded_docs, maxlen=max_length, padding='post')
	return temp_padded_docs

def convertSoftmax(output):
	if np.ndim(output) == 2:
		[output] = output

	# get top give indexes and sort them
	ind = np.argpartition(output, -3)[-3:]
	ind = ind[np.argsort(output[ind])]
	# protocols in descending order
	protos = le_proto.inverse_transform(ind)
	# probability
	probs = (output[ind])*100

	return protos, probs

def loadModel():
	return  load_model('proto_model.h5')

