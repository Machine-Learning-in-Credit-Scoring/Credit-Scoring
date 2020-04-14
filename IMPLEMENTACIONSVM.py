#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:57:26 2020

@author: valenyusty
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


plt.rc("font", size=14)

#dfsvm = pd.read_excel('lcorre.xlsx', sheet_name='Sheet1')
dfsvm = pd.read_excel('outputAN.xlsx', sheet_name='Sheet2')

h11=['Status of existing checking account', 'Duration in month',
       'Credit history', 'Purpose', 'Credit amount', 'Savings account/bonds',
       'Present employment since',
       'Installment rate in percentage of disposable income',
       'Personal status and sex', 'Other debtors / guarantors',
       'Present residence since', 'Property', 'Age in years',
       'Other installment plans ', 'Housing',
       'Number of existing credits at this bank', 'Job',
       'Number of people being liable to provide maintenance for', 'Telephone',
       'foreign worker',"Housing2"]


dfsvm['Housing2']=dfsvm['Housing2'].replace(to_replace = 2, value =-1)

dfsvm['Receive_NotReceiveCredit']=dfsvm['Receive_NotReceiveCredit'].replace(to_replace = 2, value =-1)

train, test = train_test_split(dfsvm, test_size=0.20)

train_x=train[h11]
train_y=train['Receive_NotReceiveCredit'].to_frame()

test_x=test[h11]
test_y=test['Receive_NotReceiveCredit'].to_frame()


# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(train_x, train_y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
    

model = SVC(class_weight= "balanced", kernel='rbf', gamma=10, probability=True )

# fit the model with the training data
model.fit(train_x,train_y)

# predict the target on the train dataset
predict_train = model.predict(train_x)
#print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(test_x)
#print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_y,predict_test)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(test_y,predict_test))

