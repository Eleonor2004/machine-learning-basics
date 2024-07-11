# Machine learning basics
Presently I am reading the MIT textbook: deeplearningbook and to accompany my learning i decided to do some fex projects to accompany my learning. I am just from finishing Part1: The fundamentals for deeplearning and machine learning basics.
So this repository contains some algorithms of machine learning I implemented using the knowledge af machine learning I acquired.

# Steps for a good Machine Learning algorithm
# 1- Preparation and seperation of dataset
I learnt that 80% of the effectiveness of a good machine learning algorithm depends on the quality of a dataset, so as a future datascientist and ML engineer, the collection of data is crucial. 
For now I decided to use already prepared datsets that were contained in the scikit learn library and also from the site on the link: https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/

After having the data we have to separate it into 2 parts: train set (80%) and test set (20%).
from that we constitute our design matrix: which contains the features for each sample of our dataset and a vector y which contains the labels of each sample in the cas of supervised learning.
I used the function # train_test_split found in the module sklearn.model_selection

# 2- Training of the model
We have to choose the good model for the training based on the dataset and the problem we wish to resolve. And then the model is trained using the train set(X_train, y_train). This was achieved using the algorithms of the scikitlearn library: this is a very diversed library.

# 3- Evaluation of the model
Here we use a variety of metrics such as:
- Accuracy
- F1 score
- Confusion matrix
-...
each of these metrics can be achieved using the metrics of the module sklearn.metrivs (as i said previously, it is very diversed and one of the best libraries for Machine learning)

# 4- Cross validation 
It can either be done before the evaluation or after the evaluation depending on the metrics obtained. As good data scientists, the aim is to produce a model with the best possible scores. 
Cross validation aims at finding the best values of the hyperparameters of the model choosen for example the number of neighbors in the algorithm KneighborsClassifier.
For this, we need to seperate the train set into the one that is actually used for the training and another one called the validation set. Seperating the data set for the evaluation of the hyperparameters enables us to avoid overfitting.
To find the best values of these hyperparameters, we use the following functions:
- cross_val_score : enables us to obtain the score of a model for different values of a single parameter
- validation_curve: Similar to cross_val_score , and draws the curve from the values obtained: still for a single hyperparameter
- GridSearchCV: It enables to evaluate the best set of values of hyperparameters at the time for a single model
each of these funtions are imported from the module: sklearn.model_selection
To do the cross-validation many techniques exist for the sepearation of train set from the validation set depending on the dataset, we have;
- KFold
- LeaveOneOut
- ShuffleSplit
- Stratified_KFold: most ued
- Group KFold
- Group ShuffleSplit
 NB:Cross validation is mostly used in machine learning and not in deep learning. Because in Deeplearning the datasets are very large and are diverisfied enough. More over, cross validation implies traing the model nulerous times and that will require too much ressources and time for a deep learning algorithm.

# 5- Is more data needed for the training of the algorithm ?
The answer to this question is obtained by tracing the learning_curve: which is a curve representing the score of the algorithm against the amount of data used for the training.
This function is still in the scikit learn library.

# Implemntations i did based on this knowledge:
# 1- iris_classification.py
Here i imported a datset from the link given above and i trained 4 different models and then i compared their metrics:
- DecisionTreeClassifier
- GaussianNB
- LogisticRegression
- DictVectorizer

# 2- iris_classification_sklearn.ipynb
Here i followed all of the 5 steps described above but using only one algorithm: KNeighborsClassifier.
# 3- wine_quality.py
here the objective was to evaluate the quaity based on a set of 13 features from a dataset.

