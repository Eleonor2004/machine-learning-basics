from ucimlrepo import fetch_ucirepo 
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
#Seperation of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)

#Training 
model = DecisionTreeClassifier(random_state=0)
#model = GaussianNB()
model.fit(X_train, y_train)
score = model.score(X_train,y_train)
print("The score is: ", score)
predictions = model.predict(X_test)
print("Accuracy of  the model: ",
      accuracy_score(y_test, predictions))
print("Precision of the model: ",
      precision_score(y_test, predictions, average='weighted'))
print("Recall of the model: ",
      recall_score(y_test, predictions, average='weighted'))
print("F1-Score of the model: ",
      f1_score(y_test, predictions, average='weighted'))

