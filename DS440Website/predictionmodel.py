#Import packages
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Load in Data
df = pd.read_csv('C:\\Users\\frist\\OneDrive\\Desktop\\DS440Website\\DiabetesData\\diabetes.csv')
#print(diabetesdata.head())
X = df.drop(columns=['Outcome', 'DiabetesPedigreeFunction'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#SVM Model
from sklearn.svm import SVC
SVCModel = SVC(probability=True).fit(X_train, y_train)
predictions = SVCModel.predict(X_test)
results = accuracy_score(y_test, predictions)
#print(results)

#Save the Model
filename = 'FinalSVCModel.sav'
pickle.dump(SVCModel, open(filename, 'wb'))


#tesst = SVCModel.predict_proba(X_test)
#print(tesst)
#KNN Model
#from sklearn.neighbors import KNeighborsClassifier
#KNNModel = KNeighborsClassifier().fit(X_train, y_train)
#Save the Model
#filename = 'FinalKNNModel.sav'
#pickle.dump(KNNModel, open(filename, 'wb'))

#predictions = KNNModel.predict(X_test)
#results = accuracy_score(y_test, predictions)
#print(results)

