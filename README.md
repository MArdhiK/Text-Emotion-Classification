# Text-Emotion-Classification

I got the code from kaggle:
https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

You can check this WebApps
https://mardhik-text-emotion-classification-app-hozlh5.streamlitapp.com/

This is about NLP classification, in this github I made several models   
With Deep Learning used LSTM and I got accuracy 91.40%   
With Classic Machine Learning   
    -Logistic Regresion Accuracy : 86.45%   
    -Decision Tree Accuracy      : 86.15%   
    -Light GBM Accuracy          : 88.3% 

And then I deploy my classic machine learning model on Streamlit,   
but I used Logistic Regression model,    
I have some problem with Light GBM,    
it says that 'Estimator not fitted, call fit before exploiting the model.',     
eventhough I had fitted it before, you can check it in my notebook
