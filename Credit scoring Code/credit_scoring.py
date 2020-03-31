#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load in our libraries
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pandas.api.types import is_string_dtype
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels


# # PREPROCESSING

# In[2]:


dataset = pd.read_csv("../dataset/estadistical.csv")


# In[3]:


x = dataset.drop("Receive/ Not receive credit ",axis=1)
y = dataset["Receive/ Not receive credit "]


# In[4]:


cat_mask = x.dtypes==object

cat_cols = x.columns[cat_mask].tolist()


# In[5]:


le = preprocessing.LabelEncoder()

x[cat_cols] = x[cat_cols].apply(lambda col: le.fit_transform(col))


# In[6]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, stratify = y)


# # KNN

# In[7]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(xtrain, ytrain)


# In[8]:


preKnn = neigh.predict(xtest)


# In[9]:


accuracy_score(ytest, preKnn)


# # RANDOM FOREST

# In[10]:


forest = RandomForestClassifier(max_depth=2, random_state=0)
forest.fit(xtrain, ytrain)


# In[11]:


pred_forest = forest.predict(xtest)


# In[12]:


accuracy_score(ytest,pred_forest)


# # LOGISTIC REGRESSION

# In[23]:


logRegr = LogisticRegression(random_state=0, class_weight= "balanced").fit(xtrain, ytrain)


# In[24]:


pred_logReg = logRegr.predict(xtest)


# In[25]:


accuracy_score(ytest,pred_logReg)


# In[21]:


plot_confusion_matrix(ytest, pred_logReg, classes= ytrain, normalize=False,
                      title='Confusion matrix')

plt.show()


# # SVM

# In[27]:


svm = SVC(gamma='auto',class_weight="balanced")
svm.fit(xtrain, ytrain)


# In[28]:


pred_svm = svm.predict(xtest)


# In[29]:


accuracy_score(ytest, pred_svm)


# # CONFUSION MATRIX

# In[20]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)


# In[ ]:




