import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import re
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout, Activation, Flatten, Input, concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D




def main():
    #import tokenizer
    tokenizer = pickle.load(open("tokenizer.pickle","rb"))
    vocabSize = len(tokenizer.index_word) + 1


    #create deep learning model based on our python notebook
    # Embedding
    max_features = vocabSize
    maxlen = 256
    embedding_size = 200
    # Convolution
    kernel_size = 5
    filters = 256
    pool_size = 4
    # LSTM
    lstm_output_size = 256

    model = Sequential()
    model.add(Embedding(vocabSize, embedding_size, input_length=256))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(6))
    model.add(Activation('softmax'))

    model.load_weights(r'Emotion_Recognition1.h5')


    #clean text
    def preprocess_clean(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-z0-9\s]','',text)
        return text    

    def padding(text):
        text_new = preprocess_clean(text)
        X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
        X = pad_sequences(X, maxlen=256)
        return X



    #def my_pipeline(text):
    #    text_new = preprocess_clean(text)
    #    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    #    X = pad_sequences(X, maxlen=256)
    #   return X


    ######prediction
    def predict(model_):
        clean_text = padding(text)
        loaded_model = tf.keras.models.load_model("Emotion_Recognition1.h5")
        predictions = loaded_model.predict(clean_text)
        sentiment = int(np.argmax(predictions))
        probability = max(predictions.tolist()[0])
        if sentiment==0:
            t_sentiment = "Anger"
        elif sentiment==1:
            t_sentiment = "Fear"
        elif sentiment==2:
            t_sentiment = "Joy"
        elif sentiment==3:
            t_sentiment = "Love"
        elif sentiment==4:
            t_sentiment = "Sadness"
        elif sentiment==5:
            t_sentiment = "Surprise"

        return {
            "Actual Sentence" : text,
            "Predicted Sentiment" : t_sentiment,
            "Probability" : probability
        }





    st.title("Emotion Classifier Text App")
    st.subheader("just checking")
    st.write("I dont know it either")

    raw_text = st.text_area("Place Your Text Here", max_chars = 1000)
    new_text = preprocess_clean(raw_text)
    pad = padding(new_text)

    if st.button("Classify the Text"):
        predict(model)

        







if __name__ == '__main__':
    main()
