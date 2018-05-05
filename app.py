from __future__ import division
from flask import Flask, render_template, request

import pandas as pd
from collections import Counter
import re
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)



df_all = pd.read_csv("Complete_DataSet_Clean.csv")
X_body_text = df_all.body.values
X_headline_text = df_all.headline.values
X_reliability = df_all.user_reliability.values
y = df_all.fakeness.values

tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2),max_df= 0.85, min_df= 0.01)
tfidf1 = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2),max_df= 0.85, min_df= 0.01)

X_body_tfidf = tfidf1.fit_transform(X_body_text)
X_headline_tfidf = tfidf.fit_transform (X_headline_text)

X_headline_tfidf_train, X_headline_tfidf_test, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
X_body_tfidf_train, X_body_tfidf_test, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)

lr_headline = LogisticRegression(penalty='l1')
lr_body = LogisticRegression(penalty='l1')

# train model
lr_headline.fit(X_headline_tfidf_train, y_headline_train)
lr_body.fit(X_body_tfidf_train, y_body_train)
y_headline_pred = lr_headline.predict(X_headline_tfidf_test)
y_body_pred = lr_body.predict(X_body_tfidf_test)

df_all['source_id'] = np.random.randint(1, 1000, df_all.shape[0])
df_all['likes'] = np.random.randint(1, 10000, df_all.shape[0])
df_all['dislikes'] = np.random.randint(1, 10000, df_all.shape[0])
df_all['user_reliability'] = 0
df_all['user_controversiality'] = 0

df_all.user_controversiality = df_all.user_controversiality.astype(float)
for index, row in df_all.iterrows():
        df_all.set_value(index, 'user_controversiality', (row['likes'] + row['dislikes']) ** min(row['dislikes'] / float(row['likes']) - 1, row['likes'] / float(row['dislikes']) - 1))

x4 = df_all.groupby(['source_id', 'fakeness']).size()
user_reliability = {}
df_all.user_reliability = df_all.user_reliability.astype(float)
for index, row in df_all.iterrows():
        if row['source_id'] in x4:
            df_all.set_value(index, 'user_reliability', x4[row['source_id']][0] / float(x4[row['source_id']][0] + x4[row['source_id']][1] + 0.0))
            user_reliability[row['source_id']] = x4[row['source_id']][0] / float(x4[row['source_id']][0] + x4[row['source_id']][1] + 0.0)
            #print (user_reliability[row['source_id']])
     
class cross_validation(object):
    '''This class provides cross validation of any data set why incrementally increasing number 
       of samples in the training and test set and performing KFold splits at every iteration. 
       During cross validation the metrics accuracy, recall, precision, and f1-score are recored. 
       The results of the cross validation are display on four learning curves. '''
    
    def __init__(self, model, X_data, Y_data, X_test=None, Y_test=None, 
                 n_splits=3, init_chunk_size = 1000000, chunk_spacings = 100000, average = "binary"):

        self.X, self.Y =  shuffle(X_data, Y_data, random_state=1234)
        
        
        self.model = model
        self.n_splits = n_splits
        self.chunk_size = init_chunk_size
        self.chunk_spacings = chunk_spacings        
        
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        self.X_holdout = []
        self.Y_holdout = []
        
        self.f1_train = []
        self.f1_test = []
        self.acc_train = []
        self.acc_test = []
        self.pre_train = []
        self.pre_test = []
        self.rec_train = []
        self.rec_test = []
        
        self.f1_mean_train = []
        self.f1_mean_test = []
        self.acc_mean_train = []
        self.acc_mean_test = []
        self.pre_mean_train = []
        self.pre_mean_test = []
        self.rec_mean_train = []
        self.rec_mean_test = []
        
        self.training_size = []
        self.averageType = average
    
    def make_chunks(self):
        '''Partitions data into chunks for incremental cross validation'''
        
        # get total number of points
        self.N_total = self.X.shape[0]
        # partition data into chunks for learning
        self.chunks = list(np.arange(self.chunk_size, self.N_total, self.chunk_spacings ))
        self.remainder = self.X.shape[0] - self.chunks[-1]
        self.chunks.append( self.chunks[-1] + self.remainder )



    def train_for_learning_curve(self):
        '''KFold cross validates model and records metric scores for learning curves. 
           Metrics scored are f1-score, precision, recall, and accuracy'''

        # partiton data into chunks 
        self.make_chunks()
        # for each iteration, allow the model to use 10 more samples in the training set 
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1234)
        # iterate through the first n samples
        for n_points in self.chunks: 
            
        
            # split the first n samples in k folds 
            for train_index, test_index in self.skf.split(self.X[:n_points], self.Y[:n_points]):
                self.train_index, self.test_index = train_index, test_index                
                self.X_train = self.X[self.train_index]
                self.X_test = self.X[self.test_index]
                self.Y_train = self.Y[self.train_index]
                self.Y_test = self.Y[self.test_index]
                
                self.model.fit(self.X_train, self.Y_train)
                self.y_pred_train = self.model.predict(self.X_train)
                self.y_pred_test = self.model.predict(self.X_test)
                self.log_metric_scores_()   
                
            self.log_metric_score_means_()
            self.training_size.append(n_points)
        
    def validate_for_holdout_set(self, X_holdout, Y_holdout):
        
        
        self.X_test = X_holdout
        self.Y_test = Y_holdout
        
        # partiton data into chunks 
        self.make_chunks()
        
        for n_points in self.chunks:
            
            self.X_train = self.X[:n_points]
            self.Y_train = self.Y[:n_points]

            self.model.fit(self.X_train, self.Y_train)
            self.y_pred_train = self.model.predict(self.X_train)
            self.y_pred_test = self.model.predict(self.X_test)
            self.log_metric_scores_()   

            self.log_metric_score_means_()
            self.training_size.append(n_points)
            
            
    
                            
    def log_metric_score_means_(self):
        '''Recrods the mean of the four metrics recording during training'''
        self.f1_mean_train.append(np.sum(self.f1_train)/len(self.f1_train))
        self.f1_mean_test.append(np.sum(self.f1_test)/len(self.f1_test))
        
        self.acc_mean_train.append(np.sum(self.acc_train)/len(self.acc_train))
        self.acc_mean_test.append(np.sum(self.acc_test)/len(self.acc_test))
        
        self.pre_mean_train.append(np.sum(self.pre_train)/len(self.pre_train))
        self.pre_mean_test.append(np.sum(self.pre_test)/len(self.pre_test))
        
        self.rec_mean_train.append(np.sum(self.rec_train)/len(self.rec_train))
        self.rec_mean_test.append(np.sum(self.rec_test)/len(self.rec_test))
        
        self.reinitialize_metric_lists_()
            
            
    def reinitialize_metric_lists_(self):
        '''Reinitializes metrics lists for training'''
        self.f1_train = []
        self.f1_test = []
        self.acc_train = []
        self.acc_test = []
        self.pre_train = []
        self.pre_test = []
        self.rec_train = []
        self.rec_test = []

            
    def log_metric_scores_(self):
        '''Records the metric scores during each training iteration'''
        self.f1_train.append(f1_score(self.Y_train, self.y_pred_train, average=self.averageType))
        self.acc_train.append(accuracy_score( self.Y_train, self.y_pred_train) )

        self.pre_train.append(precision_score(self.Y_train, self.y_pred_train, average=self.averageType))
        self.rec_train.append(recall_score( self.Y_train, self.y_pred_train, average=self.averageType) )

        self.f1_test.append(f1_score(self.Y_test, self.y_pred_test, average=self.averageType))
        self.acc_test.append(accuracy_score(self.Y_test, self.y_pred_test))

        self.pre_test.append(precision_score(self.Y_test, self.y_pred_test, average=self.averageType))
        self.rec_test.append(recall_score(self.Y_test, self.y_pred_test,average=self.averageType))
            

    def plot_learning_curve(self):
        '''Plots f1 and accuracy learning curves for a given model and data set'''
        
        fig = plt.figure(figsize = (17,12))
        # plot f1 score learning curve
        fig.add_subplot(221)   # left
        plt.title("F1-Score vs. Number of Training Samples")
        plt.plot(self.training_size, self.f1_mean_train, label="Train")
        plt.plot(self.training_size, self.f1_mean_test, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("F1-Score")
        plt.legend(loc=4);
        
        # plot accuracy learning curve
        fig.add_subplot(222)   # right 
        plt.title("Accuracy vs. Number of Training Samples")
        plt.plot(self.training_size, self.acc_mean_train, label="Train")
        plt.plot(self.training_size, self.acc_mean_test, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Accuracy")
        plt.legend(loc=4);
        
        # plot precision learning curve
        fig.add_subplot(223)   # left
        plt.title("Precision Score vs. Number of Training Samples")
        plt.plot(self.training_size, self.pre_mean_train, label="Train")
        plt.plot(self.training_size, self.pre_mean_test, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Precision")
        plt.ylim(min(self.pre_mean_test), max(self.pre_mean_train) + 0.05)
        plt.legend(loc=4);
        
        # plot accuracy learning curve
        fig.add_subplot(224)   # right 
        plt.title("Recall vs. Number of Training Samples")
        plt.plot(self.training_size, self.rec_mean_train, label="Train")
        plt.plot(self.training_size, self.rec_mean_test, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Recall")
        plt.legend(loc=4);
        plt.savefig('static/img/plot.png')


@app.route("/")
def main():
	return render_template('index.html')
	
@app.route("/Verify_Header",methods=['POST'])
def Verify_Header():
	read_input = pd.read_csv("data.csv")
	read_input['headline']=request.form['headline']
	source_id=request.form['source_id']  
	input_body_text = read_input.body.values
	input_headline_text = read_input.headline.values
	input_reliability = read_input.user_reliability.values
	input_y = df_all.fakeness.values
	input_headline_tfidf = tfidf.transform(input_headline_text)
	input_headline_pred = lr_headline.predict(input_headline_tfidf)
	ans=""
	if input_headline_pred[0]==0:
	 	ans="real"
	else:
	 	ans="fake"
	reliability = ""
	if int(source_id) in user_reliability:
		reliability = str(user_reliability[int(source_id)] * 100) + "%"
	else:
		reliability = "not applicable, because this news is posted by a new user."
	likes = np.random.randint(1, 10000)
	dislikes = np.random.randint(1, 10000)
	controversiality = str(100*((likes + dislikes) ** (min(likes / float(dislikes), dislikes / float(likes)) - 1))) + "%"
	output="<h1>The Learning Algorithm predicts that the news is "+ans+"<br> The reliability of the given source is "+reliability+"<br> The controversiality of the given post is "+controversiality+"</h1>"
	return output	

@app.route("/Verify_Body",methods=['POST'])
def Verify_Body():
	read_input = pd.read_csv("data.csv")
	read_input['body']=request.form['body']
	source_id=request.form['source_id']
	input_body_text = read_input.body.values
	input_headline_text = read_input.headline.values
	input_reliability = read_input.user_reliability.values
	input_y = df_all.fakeness.values
	input_body_tfidf = tfidf1.transform(input_body_text)
	input_body_pred = lr_body.predict(input_body_tfidf)
	ans=""
	if input_body_pred[0]==0:
		ans="real"
	else:
		ans="fake"
	reliability = ""
	if int(source_id) in user_reliability:
		reliability = str(user_reliability[int(source_id)] * 100) + "%"
	else:
		reliability = "not applicable, because this news is posted by a new user."
	likes = np.random.randint(1, 10000)
	dislikes = np.random.randint(1, 10000)
	controversiality = str(100*((likes + dislikes) ** (min(likes / float(dislikes), dislikes / float(likes)) - 1))) + "%"
	output="<h1>The Learning Algorithm predicts that the news is "+ans+"<br> The reliability of the given source is "+reliability+"<br> The controversiality of the given post is "+controversiality+"</h1>"
	return output	

@app.route("/Accuracy_Header_plot",methods=['POST'])
def Accuracy_Header_plot():
	xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)
	cv = cross_validation(lr_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 5000, chunk_spacings = 1000, average = "binary")
	cv.validate_for_holdout_set(xtest, ytest)
	cv.plot_learning_curve()
	return render_template('plot.html')

@app.route("/Accuracy_Header_Score",methods=['POST'])
def Accuracy_Header_Score():
    output="Logistic Regression F1 and Accuracy Scores :<br>"+"F1 score {:.4}%".format( f1_score(y_headline_test, y_headline_pred, average='macro')*100 )+"<br>Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_headline_pred)*100)
    return output

@app.route("/Accuracy_Body_plot",methods=['POST'])
def Accuracy_Body_plot():
    xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)
    cv = cross_validation(lr_body, xtrain, ytrain , n_splits=5,init_chunk_size = 5000, chunk_spacings = 1000, average = "binary")
    cv.validate_for_holdout_set(xtest, ytest)
    cv.plot_learning_curve()
    return render_template('plot.html')

@app.route("/Accuracy_Body_Score",methods=['POST'])
def Accuracy_Body_Score():
    output="Logistic Regression F1 and Accuracy Scores :<br>"+"F1 score {:.4}%".format( f1_score(y_body_test, y_body_pred, average='macro')*100 )+"<br>Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_body_pred)*100)
    return output

if __name__ == "__main__":
    app.run()