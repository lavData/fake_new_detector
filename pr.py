import string
import pandas as pd
from PIL import Image
import streamlit as st
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer

def remove_stopword(x):
     return [i.rstrip('_') for i in x if i not in stopword and i not in [',', '']]

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


new_df = pd.DataFrame({'text': [text], 'domain': [''], 'label': ['']})
new_df = pd.concat([news_df, new_df], axis=0, ignore_index=True)

puncs = string.punctuation + '\n“”‘’'
stopword = pd.Series(list(stopword)).str.replace(' ', '_').to_list()


clean_text = new_df['text'].replace(f'[{puncs}]', ',', regex=True).\
             apply(ViTokenizer.tokenize).str.lower().\
             str.split().apply(remove_stopword)

new_df['clean_text'] = clean_text
new_df['clean_text'] = new_df.apply(lambda x: ' '.join(x['clean_text']) + ' ' + x['domain'], axis=1)
train_data = TfidfVectorizer(lowercase=False).fit_transform(new_df['clean_text'])

x_train = train_data[:-1]
x_test = train_data[-1:]


if st.button('Predict it news is real or fake 👈'):


    if model == 'LogisticRegressionCV':
        lg_re = LogisticRegressionCV(Cs=20, cv=5, solver='newton-cg', max_iter=10000).\
                fit(x_train, news_df.label)
        result = lg_re.predict(x_test)
        st.write(result)
        if result[0] == 0:
            st.write('fake new')
        else:
            st.write('real new')
        
    elif model == 'DecisionTreeClassifier':
        tree = DecisionTreeClassifier(random_state=42, max_features=None, max_leaf_nodes=30).\
                     fit(x_train, news_df.label)
        result = tree.predict(x_test)
        if result[0] == 0:
            st.write('fake new')
        else:
            st.write('real new')
    else:
        MLPC = MLPClassifier(hidden_layer_sizes=(50), alpha=0, solver='lbfgs', max_iter=10000).\
               fit(x_train, news_df.label)
        result = MLPC.predict(x_test)
        if result[0] == 0:
            st.write('fake new')
        else:
            st.write('real new')



