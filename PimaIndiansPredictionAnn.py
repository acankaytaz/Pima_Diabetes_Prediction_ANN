# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:43:38 2020

@author: acan
"""
#1. kutuphaneler
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#2.1. Veri Yukleme
veriler = pd.read_csv('pima_diabetes.csv')
print(veriler.shape)

"""
#verileri ayırma ver1:
Y = np.array(veriler['Outcome'])
del veriler['Outcome']
X = np.array(veriler)
"""

#veri ayırma ver2:
X = veriler.iloc[:,0:8].values
Y = veriler.iloc[:,8].values

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0) #0.33 ile .66 olarak verileri böldük

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#3 Yapay Sinir agı
import keras
from keras.models import Sequential 
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(64, init = 'uniform', activation = 'relu' , input_dim = 8))  #giriste 8 noron(input dim) gizli katmanı 64 noron

classifier.add(Dense(32, init = 'uniform', activation = 'relu')) #2. gizli katman(32 noron)

classifier.add(Dense(32, init = 'uniform', activation = 'relu')) #3. gizli katman(32 noron)

classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid')) #cikis katmanı

classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

classifier.fit(X_train, y_train, epochs=150, batch_size=10) 
#epoch = öğrenme algoritmasının kac kere calisacagi, batch = bir iterasyonda üzerinde calisilacak numune sayisi..

y_pred = classifier.predict(X_test)

y_pred = (y_pred >= 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #test ve tahmin verilerini karsilastirdik
print("---------")
print(cm)
print("---------")

#print(classifier.summary())

