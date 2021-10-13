# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import seaborn as sn
from sklearn.metrics import confusion_matrix
import re
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import string
from string import punctuation

def processMessage(tweet):
    tweet = re.sub(r'\&\w*;','',tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub(r'\$\w*', '',tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\.*\/\w*', '', tweet)
    tweet = re.sub(r'#\w*', '', tweet)
    tweet = re.sub(r'[' + punctuation.replace('@','') + ']+', ' ',tweet)
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    tweet = re.sub(r'\s\s+', '',' ', tweet)
    tweet = tweet.lstrip(' ')
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet

class NaiveBayes:
    accuracy =0
    precission = 0
    feature = 0
    recall = 0
    imagePath = ''
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
    label_encoder = LabelEncoder()
    
    def trainModel(self,filepath):
        print('Inside trainModel')
        df_inputdata=pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\New folder\\SpamHam.csv',usecols =[0,1],encoding='latin-1')
        #print(df_inputData.head())
        df_inputdata.rename(columns={'v1':'Category','v2':'Message'}, inplace = True)
        #df_inputdata.head()
        df_inputdata['Message'] = df_inputdata['Message'].apply(processMessage)
        #convert the labels text to numbers
        NaiveBayes.label_encoder = preprocessing.LabelEncoder()
        NaiveBayes.label_encoder.fit_transform(df_inputdata['Category'])
        df_inputdata['Category'] = NaiveBayes.label_encoder.transform(df_inputdata['Category'])
        X=df_inputdata.Message
        y=df_inputdata.Category
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)
        NaiveBayes.tfidf_vect.fit(X_train)
        xtrain_tfdif = NaiveBayes. tfidf_vect.transform(X_train)
        xvalid_tfidf =NaiveBayes. tfidf_vect.transform(X_test)
        model = naive_bayes.MultinominalNB()
        model.fit(xtrain_tfdif,y_train)
        y_pred = model.predict(xvalid_tfidf)
        NaiveBayes.accuracy = metrics.accuracy_score(y_test, y_pred)
        #save the trained model into hard disk
        modelfilename = 'NaiveBayesModel.sav'
        pickle.dump(model, open(modelfilename, 'wb'))
        #confusion matrix
        #get the confusion matrix
        cm=confusion_matrix(y_test,y_pred)
        conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
        #plt.figure(figsize = (8,5))
        #sn.heatmap(conf_matrix,annot=True,fmt='d',cmap="YlGnBu")
        #imagefile='NaiveBayes_confusion.jpg'
        #plt.savefig(imagefile)
        #performance parameters
        imagePath = ''
        print('Image fil path', imagePath)
        
        NaiveBayes.precission = metrics.precission_score(y_test,y_pred,average=None)
        NaiveBayes.recall= metrics.recall_score(y_test, y_pred, average=None)
        
        print('NaiveBayes Model', 'Training Completed')
    def getAccuracy(self):
        return NaiveBayes.accuracy
    def getPerfmatrix(self):
        return NaiveBayes.precision, NaiveBayes.recall, NaiveBayes.fmeasure, NaiveBayes.imagePath
    def getPrediction(self, inputTweet):
        myData = np.array([inputTweet])
        myData = NaiveBayes.tfidf_vect.transform(myData)
        filename= 'NaiveBayesModel.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(myData)
        print(y_pred)
        vals = NaiveBayes.label_encoder.inverse_transform([y_pred[0]])
        print(vals[0])
        return vals[0]