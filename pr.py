import string
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


image = Image.open('./image.jpeg')
st.image(image)
st.title('ENGLISH FAKE NEW DETECTOR')

model =  st.selectbox('Model you want',['LogisticRegressionCV',\
                                        'KNeighborsClassifier',\
                                        'MLPClassifier'])
text =  st.text_area('Piece of news here')
news_df = pd.read_csv('./vn_news_223_tdlfr.csv', encoding='utf-8')
stopword = open('./vietnamese-stopwords.txt', 'r',encoding='utf-8')
stopword = set([i.rstrip() for i in stopword.readlines()])

def preprocessing(text):
    chars_to_remove = list(string.punctuation)
    chars_to_remove.extend(['“', '”'])
    
    no_punc = ''.join([char for char in text if char not in chars_to_remove])
    no_punc = no_punc.replace('\n', ' ')
    
    tokenize = ViTokenizer.tokenize(no_punc)
    
    clean_words = [word.lower() for word in tokenize.split() if word.lower() not in stopword]
    return clean_words

vectorizer = TfidfVectorizer(analyzer=preprocessing)

new_df = pd.DataFrame({'text': [text], 'domain': [''], 'label': ['']})
new_df = pd.concat([news_df, new_df], axis=0, ignore_index=True)
train_data = vectorizer.fit_transform(new_df['text'])
x_train = train_data[:-1]
x_test = train_data[-1:]


if st.button('Predict it news is real or fake 👈'):


    if model == 'LogisticRegressionCV':
        lg_re = LogisticRegressionCV(Cs=20, cv=5, solver='lbfgs', max_iter=10000).fit(x_train, news_df.label)
        result = lg_re.predict(x_test)
        if result[0] == 0:
            st.write('fake new')
        else:
            st.write('real new')
        
    elif model == 'KNeighborsClassifier':
        k_neighbor = KNeighborsClassifier(n_neighbors=6, algorithm='auto').fit(x_train, news_df.label)
        result = k_neighbor.predict(x_test)
        if result[0] == 0:
            st.write('fake new')
        else:
            st.write('real new')
    else:
        MLPC = MLPClassifier(hidden_layer_sizes=(50), alpha=0, solver='lbfgs', max_iter=10000).fit(x_train, news_df.label)
        result = MLPC.predict(x_test)
        if result[0] == 0:
            st.write('fake new')
        else:
            st.write('real new')



