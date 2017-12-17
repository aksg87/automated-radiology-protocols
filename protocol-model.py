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

from copy import deepcopy
from string import punctuation
from random import shuffle
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def ingest():
	data = pd.read_csv("./65k lines.csv")

	# data['Diagnosis'] = data['Diagnosis'].str.replace("\[(.*?)\]", "", case=False)
	print("diagnosis codes", data['Diagnosis'])

	le = preprocessing.LabelEncoder()
	le.fit(data['Protocol'])
	print("classes", list(le.classes_))

	encoded_P = le.transform(data['Protocol'])
	encoded_P = np_utils.to_categorical(encoded_P)
	print("one hot", data['Protocol'])

	return data['Diagnosis'], encoded_P

def tokenize(text):
        # text = unicode(text.decode('utf-8').lower())
        tokens = tokenizer.tokenize(text)
	    # tokens = filter(lambda t: not t.startswith('@'), tokens)
        return tokens

def postprocess(data):
    # data = data.head(n)
    tokens = data.progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    # data.reset_index(inplace=True)
    # data.drop('index', inplace=True, axis=1)
    return tokens


def labelizeText(text, label_type):
    labelized = []
    for i,v in tqdm(enumerate(text)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


diagnosis, protocols = ingest()

tokens = postprocess(diagnosis)

x_train, x_test, y_train, y_test = train_test_split(np.array(tokens),
                                                    np.array(protocols), test_size=0.2)

x_train = labelizeText(x_train, 'TRAIN')
x_test = labelizeText(x_test, 'TEST')

text_w2v = Word2Vec(size=200, min_count=10)
text_w2v.build_vocab([x.words for x in tqdm(x_train)])
text_w2v.train([x.words for x in tqdm(x_train)], total_examples=text_w2v.corpus_count, epochs=text_w2v.iter)

print(text_w2v.most_similar('dyspnea'))

