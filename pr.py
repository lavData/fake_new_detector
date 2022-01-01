import streamlit as st
from PIL import Image
image = Image.open('./image.jpeg')
st.image(image)
st.title('ENGLISH FAKE NEW DETECTOR')
st.selectbox('Model you want', ['linear' , 'clsssiffyfyfy'])
st.text_area('Piece of news here')
if st.button('Predict it news is real or fake 👈'):
    st.write('real.')