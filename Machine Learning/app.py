import streamlit as st
import altair as alt
import plotly.express as px

import pandas as pd
import numpy as np

import joblib



pipe_lr = joblib.load(open("nlp_classicML(lr).pkl","rb"))


def predict_text(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def predict_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results


emotion_dict = {"anger":"😠", "fear":"😨😱", "joy":"😂",
				"love":"😍", "sadness":"😔", "surprise":"😮"}


def main():
	st.title("Welcome to Emotion Text Classifier")
	st.image("textemotion.png")
	st.write("**In this WebApps, I make Emotion Text Classifier with Classic Machine Learning**")
	st.write("**In this WebApps, I use Logistic Regression model**")
	
	#st.subheader("Emotion in Text")

	with st.form(key='emotion_clf_form'):
		raw_text = st.text_area("Type your text here")
		submit_text = st.form_submit_button(label ='Submit')

	if submit_text :
		col1, col2 = st.columns(2)

		prediction = predict_text(raw_text)
		probability = predict_proba(raw_text)

		with col1:
			st.success("Original Text")
			st.write(raw_text)

			st.success("Prediction")
			emoji_icon = emotion_dict[prediction]
			st.write("{}:{}".format(prediction, emoji_icon))
			st.write("Confidence:{}".format(np.max(probability)))

		with col2:
			st.success("Prediction Probaility")
			proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
			proba_df_clean = proba_df.T.reset_index()
			proba_df_clean.columns = ["Emotions", "Probability"]

			fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotions',y='Probability',color='Emotions')
			st.altair_chart(fig, use_container_width=True)







if __name__ == '__main__':
	main()