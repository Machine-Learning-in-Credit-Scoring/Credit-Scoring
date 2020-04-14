#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:16:23 2020

@author: valenyusty
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

df = pd.read_excel('outputAN.xlsx', sheet_name='Sheet1')

h11=['Status of existing checking account', 'Duration in month',
       'Credit history', 'Savings account/bonds',
       'Installment rate in percentage of disposable income']


df['Receive_NotReceiveCredit']=df['Receive_NotReceiveCredit'].replace(to_replace = 2, value =0) 

train, test = train_test_split(df, test_size=0.2)


train_x = train[h11]
X_scaled = preprocessing.scale(train_x)
train_y = train['Receive_NotReceiveCredit'].to_frame()

test_x = test[h11]
X_scaled2 = preprocessing.scale(test_x)
test_y = test['Receive_NotReceiveCredit'].to_frame()


import statsmodels.api as sm
logit_model=sm.Logit(train_y,train_x)
result=logit_model.fit()
print(result.summary2())

# Luego quitas el p mas grande 

model = LogisticRegression(penalty='l2', C=0.1, class_weight= "balanced")


# fit the model with the training data

model.fit(X_scaled,train_y)

## coefficeints of the trained model
print('Coefficient of model :', model.coef_)
#
## intercept of the model
print('Intercept of model',model.intercept_)

# predict the target on the train dataset
predict_train = model.predict(train_x)
#print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(X_scaled2)
#print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_y,predict_test)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(test_y,predict_test))

#ROC 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_y, model.predict(X_scaled2))
fpr, tpr, thresholds = roc_curve(test_y, model.predict_proba(X_scaled2)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = 0.71)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()