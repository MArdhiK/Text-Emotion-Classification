import streamlit as st

import pandas as pd
import numpy as np

import seaborn as sns

import joblib
import pickle

model = pickle.load(open("tokenizer.pickle","rb"))

def predict_emotions(docx):
	results = model.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = model.predict_proba([docx])
	return results

emotions_emoji_dict = {"joy":"😂", "sadness":"😔","anger":"😠", "fear":"😨😱", "love":"😍"   ,"surprise":"😮"}


def main():
	st.title("Emotion Classifier Text App")
	menu = ["Home", "Monitor", "About"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Home":
		st.subheader("Home Emotion Classification Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Text Here")
			submit_text = st.form_submit_button(label='Submit')
			
		if submit_text:
			col1, col2 = st.columns(2)

			#apply function here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction, emoji_icon))

			with col2:
				st.success("Prediction Probability")
				st.write(probability)



	elif choice == "Monitor":
		st.subheader("Monitor App")


	else:
		st.subheader("About")





if __name__ == '__main__':
	main()