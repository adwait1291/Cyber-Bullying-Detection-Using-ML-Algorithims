#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:22:51 2021

@author: adwaitkesharwani
"""
import numpy as np
import pandas as pd
import pickle
import base64
import streamlit as st 
import string
from nltk import pos_tag
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#---------------Functions to clean the input text--------------#

def tokenize_remove_punctuation(text):
        clean_text = []
        text = text.split(" ")
        for word in text:
            word = list(word)
            new_word = []
            for c in word:
                if c not in string.punctuation:
                    new_word.append(c)
                word = "".join(new_word)
            clean_text.append(word)
        return clean_text
    
def remove_stopwords(text):
        clean_text = []
        for word in text:
            if word not in stopwords:
                clean_text.append(word)
        return clean_text
    
def pos_tagging(text):
   tagged = nltk.pos_tag(text)
   return tagged
            
def get_wordnet(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
def clean_text(text):
        text = str(text)
            #Converting text to lower-case
        text = text.lower()
            #tokenize and remove punctuations from the text
        text = tokenize_remove_punctuation(text)
            #remove words containing numericals
        text = [word for word in text if not any(c.isdigit() for c in word)]
            #remove stopwords
        text = remove_stopwords(text)
            #remove empty tokens
        text = [ t for t in text if len(t) > 0]
            #pos tagging
        pos_tags = pos_tagging(text)
            #Lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0],get_wordnet(t[1])) for t in pos_tags]
            #remove words with only one letter
        text = [ t for t in text if len(t)>1]
            #join all words
        text = " ".join(text)
        return text
        
def transform_input(self, text):
        text = np.array([text])
        text = pd.Series(text)
        text = clean_text(text)
        return text

#---------------Loading trained models--------------#


pickle_in = open("vectorizer.pkl","rb")
vect=pickle.load(pickle_in)

pickle_in = open("LinearSVC.pkl","rb")
LinearSVC = pickle.load(pickle_in)

pickle_in = open("MultinomialNB.pkl","rb")
MultinomialNB = pickle.load(pickle_in)

pickle_in = open("LogisticRegression.pkl","rb")
LogisticRegression = pickle.load(pickle_in)

pickle_in = open("KNeighborsClassifier.pkl","rb")
KNeighborsClassifier = pickle.load(pickle_in)

#---------------HTML--------------#

html_temp = """
    <div style="background-color: pink;padding:10px; border-radius: 5px;
    height: 100%;
    width: 100%;">
    <h2 style="color:white;text-align:center;">Cyber Bullying Detector</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)


#---------------Creating selectbox--------------#

option = st.selectbox(
     'Select Model',
    ('Select','Linear SVC', 'K-Nearest Neighbour', 'Logistic Regression', 'Multinomial Naive Bayes'))


#---------------Input--------------#

comment = st.text_input("Enter any comment"," ")
comment = clean_text(comment)
comment = np.array([comment])
comment = pd.Series(comment)
comment =vect.transform(comment)
pred = [0]# creating a list to store the output value
     

#---------------Predicting output--------------#

if st.button("Predict"):
        if option == 'MultinomialNB':
                pred = MultinomialNB.predict(comment)
                if(pred[0] == 1):
                   st.success('prediction: {}'.format("Bullying comment!!!!"))
                else:
                   st.success('prediction: {}'.format("Normal comment."))

        elif option == 'LinearSVC':
                pred = LinearSVC.predict(comment)
                if(pred[0] == 1):
                    st.success('prediction: {}'.format("Bullying comment!!!!"))
                else:
                    st.success('prediction: {}'.format("Normal comment."))

        elif option == 'KNN':
                pred = KNeighborsClassifier.predict(comment)
                if(pred[0] == 1):
                    st.success('prediction: {}'.format("Bullying comment!!!!"))
                else:
                    st.success('prediction: {}'.format("Normal comment."))

        elif option == 'Logistic Regression':
                if(pred[0] == 1):
                    st.success('prediction: {}'.format("Bullying comment!!!!"))
                else:
                    st.success('prediction: {}'.format("Normal comment."))
        else:
                st.write("You haven't selected any model :(")





