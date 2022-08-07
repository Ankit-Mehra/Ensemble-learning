# -*- coding: utf-8 -*-
"""
Created on TUE July 26, 2022
Author-Ankit Mehra 301154845
Ensemble learning
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import seaborn as sns

# Load & check the data:    #####

# 1.Load the data (pima-indians-diabetes.csv)  into a pandas dataframe named
# df_firstname where first name is you name.

df_ankit = pd.read_csv('pima-indians-diabetes.csv')

# 2.	Add the column names i.e. add a header record.
df_ankit.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class']

# 3.	Carryout some initial investigations:
# a.	Check the names and types of columns.
print(df_ankit.columns)
print(df_ankit.dtypes)
# b.	Check the missing values.
df_ankit.isna().sum()
df_ankit.isnull().sum()
# c.	Check the statistics of the numeric fields (mean, min, max, median, count,..etc.)
df_ankit.describe()
# d.	Check the categorical values, if any.
df_ankit.nunique()
df_ankit.info()
df_ankit.select_dtypes(include=['object']).nunique(axis=0)


# f.	Print out the total number of instances in each class and note into your report and
#       explain your findings in terms of balanced and un-balanced.
def show_instance(data):
    for column in data.columns:
        print(data[column].value_counts())
        print("==============")


show_instance(df_ankit)

# Pre-process and prepare the data for machine learning  #####

# 4.	Prepare a standard scaler transformer to transform all the numeric
# values. Name the transformer transformer_firstname.
transformer_ankit = StandardScaler()

# 5.	Split the features from the class.

X = df_ankit.drop(['Class'], axis='columns')
y = df_ankit['Class']

# 6.	Split your data into train 70% train and 30% test, use 42 for the seed.
# Name the train/test dataframes as follows : X_train_firstname, X_test firstname
# , y_train firstname, y_test firstname.

X_train_Ankit, X_test_Ankit, y_train_Ankit, y_test_Ankit = train_test_split(X, y,
                                                                            test_size=0.3,
                                                                            random_state=42)
# 7.	Apply fit, transform the transformer prepared in step 4 to the features i.e.
# X_train_firstname, X_test firstname.
X_train_Ankit_std = transformer_ankit.fit_transform(X_train_Ankit)
X_test_Ankit_std = transformer_ankit.fit_transform(X_test_Ankit)

# Exercise #1 :Hard voting   ######

# 8.	Define 5 classifiers & give them names of your choice, just add at the
# end _X, where X is the first letter of your last name. Details of classifiers
# are as follows:
# a.	 Logistic Regression set max_iter=1400
# b.	 Random Forest Classifier, use the defaults
# c.	 Support vector machines use the defaults
# d.	 Decision Tree Classifier set criterion="entropy" & max_depth =42
# e.	Extra Trees Classifier, use the defaults


log_clf_M = LogisticRegression(random_state=42, max_iter=1400)
rnd_clf_M = RandomForestClassifier(random_state=42)
svm_clf_M = SVC(random_state=42)
dtree_clf_M = DecisionTreeClassifier(criterion="entropy",
                                     max_depth=42, random_state=42)
extra_tree_clf_M = ExtraTreesClassifier(random_state=42)

# 9.	Define a voting classifier that contains all the above classifiers
# as estimators, set the voting to hard.
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf_M), ('rf', rnd_clf_M), ('svc', svm_clf_M),
                ('dtree', dtree_clf_M), ('extra_tree', extra_tree_clf_M)],
    voting='hard')

# 10.	Fit the training data to the voting classifier and predict
# the first three instances of test data.
voting_clf.fit(X_train_Ankit_std, y_train_Ankit)
y_pred_hard = voting_clf.predict(X_train_Ankit_std[0:3])
print(y_pred_hard)

# 11.	Print out for each classifier (including the voting classifier)
# and for each instance the predicted and the actual values and note
# them in your written response.
for clf in (log_clf_M, rnd_clf_M, svm_clf_M, dtree_clf_M, extra_tree_clf_M, voting_clf):
    clf.fit(X_train_Ankit_std, y_train_Ankit)
    y_pred = clf.predict(X_train_Ankit_std[0:3])
    print(f"{clf.__class__.__name__}\n")
    dictData = {'Actual Values': y_test_Ankit.iloc[0:3], 'Predicted Value': y_pred}
    print(f"{pd.DataFrame(data=dictData)}\n====================")

# Exercise #2: Soft voting   ######

# 	Define 5 classifiers & give them names of your choice, just add at the
# end _X, where X is the first letter of your last name. Details of classifiers
# are as follows:
# a.	 Logistic Regression set max_iter=1400
# b.	 Random Forest Classifier, use the defaults
# c.	 Support vector machines use the defaults
# d.	 Decision Tree Classifier set criterion="entropy" & max_depth =42
# e.	Extra Trees Classifier, use the defaults


log_clf_soft_M = LogisticRegression(random_state=42, max_iter=1400)
rnd_clf_soft_M = RandomForestClassifier(random_state=42)
svm_clf_soft_M = SVC(random_state=42, probability=True)
dtree_clf_soft_M = DecisionTreeClassifier(criterion="entropy",
                                          max_depth=42, random_state=42)
extra_tree_clf_soft_M = ExtraTreesClassifier(random_state=42)

# .	Define a voting classifier that contains all the above classifiers
# as estimators, set the voting to hard.
voting_clf_soft_M = VotingClassifier(
    estimators=[('lr_soft', log_clf_soft_M),
                ('rf_soft', rnd_clf_soft_M),
                ('svc_soft', svm_clf_soft_M),
                ('dtree_soft', dtree_clf_soft_M),
                ('extra_tree_soft', extra_tree_clf_soft_M)],
    voting='soft')

# .	Fit the training data to the voting classifier and predict
# the first three instances of test data.
voting_clf_soft_M.fit(X_train_Ankit_std, y_train_Ankit)
y_pred_soft = voting_clf.predict(X_test_Ankit[0:3])
print(y_pred_soft)

# .	Print out for each classifier (including the voting classifier)
# and for each instance the predicted and the actual values and note
# them in your written response.
for clf in (log_clf_M, rnd_clf_soft_M, svm_clf_soft_M, dtree_clf_soft_M, extra_tree_clf_soft_M, voting_clf_soft_M):
    clf.fit(X_train_Ankit_std, y_train_Ankit)
    y_pred = clf.predict(X_train_Ankit_std[0:3])
    print(f"{clf.__class__.__name__}\n")
    dictData = {'Actual Values': y_test_Ankit.iloc[0:3], 'Predicted Value': y_pred}
    print(f"{pd.DataFrame(data=dictData)}\n====================")

# Exercise #3: Random forests & Extra Trees   (35 marks) #######

# 15.	Building on the previous classifiers you defined, create two different
# pipelines as follows:
# a.	Pipeline #1 : Name it pipeline1_firstname. The pipeline should have
# two steps, the first the transformer you prepared in step #4 and the second
# the Extra Trees Classifier you prepared in step 8.e.

pipeline1_Ankit = Pipeline([
    ('std_scalar', transformer_ankit),
    ('extra_tree', extra_tree_clf_soft_M)
])

# b.	Pipeline #2 : Name it pipeline1_firstname. The pipeline should have two steps
# the first the transformer you prepared in step #4 and the second the Decision Tree
# Classifier you prepared in step 8.d.
pipeline2_Ankit = Pipeline([
    ('std_scalar', transformer_ankit),
    ('dtree', dtree_clf_soft_M)
])

# 16.	Fit the original data to both pipelines.

pipeline1_Ankit.fit(X_train_Ankit, y_train_Ankit)
pipeline2_Ankit.fit(X_train_Ankit, y_train_Ankit)

# 17.	Carry out a 10-fold cross validation for both pipelines set shuffling to true and
# random_state to 42
cross_validation = KFold(n_splits=10, shuffle=True, random_state=42)
scores1 = cross_val_score(pipeline1_Ankit, X_train_Ankit, y_train_Ankit, cv=cross_validation)
scores2 = cross_val_score(pipeline2_Ankit, X_train_Ankit, y_train_Ankit, cv=cross_validation)
print(scores1)
print(scores2)

# 18.	Printout the mean score evaluation for both pipelines, note the final results
# in your written response and advise which pipeline performed better.
print(np.mean(scores1))
print(np.mean(scores2))


# 19.	Predict the test using both pipelines and printout the confusion matrix, precision,
# recall and accuracy scores and record the results in your written response. (Use a loop)
pipe1_pred = pipeline1_Ankit.predict(X_test_Ankit)
pipe2_pred = pipeline2_Ankit.predict(X_test_Ankit)


# function to print different scores
def print_test_scores(pred, y_true):
    clf_report = pd.DataFrame(classification_report(y_true, pred, output_dict=True))
    print("Test Result:\n=====================")
    print(f"Accuracy Score: {accuracy_score(y_true, pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_true, pred)}\n")


# Function for Plotting Heat map/Confusion Matrix
def plot_heat_map(pred, y_true):
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(confusion_matrix(pred, y_true), cmap='Blues', annot=True, fmt='g')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    ax.set_title(f"Confusion Matrix for test Cases")
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()


print("=========Extra Trees============")
print_test_scores(pipe1_pred, y_test_Ankit)
print("=========Decision Tree============")
print_test_scores(pipe2_pred, y_test_Ankit)

plot_heat_map(pipe1_pred, y_test_Ankit)
plot_heat_map(pipe2_pred, y_test_Ankit)

# 21.	Carryout a randomized grid search on the first Pipeline you defined in step #15.
# The randomized grid search should investigate the use of different values for the
# following two parameters:
# a.	The number of trees in the forest. Use a distribution between the
# values of 10 trees and 3000 trees with a step of 20.
# b.	The maximum depth of the tree.  Use a distribution between the
# values of 1 max depth and 1000 max_depth with a step of 2.

tree_distribution = {
    'extra_tree__n_estimators': [i for i in range(10, 3000, 20)],
    'extra_tree__max_depth': [i for i in range(1, 1000, 2)]
}

random_search = RandomizedSearchCV(pipeline1_Ankit,
                                   param_distributions=tree_distribution,
                                   n_iter=10, cv=5,
                                   random_state=42,
                                   scoring='accuracy')

# 22.	Fit your training data to the randomized gird search object.
random_search.fit(X_train_Ankit, y_train_Ankit)

# 23.	Print out the best parameters and accuracy score for randomized grid search
# and record them in your written response and comment on these results.
# For example, compare the accuracy score to the one you obtained in
# step # 17 and compare the parameters suggested by the randomized grid search
# to the ones you used in the classifier setup i.e. step #8. list any conclusions you noticed.
print(f"Best Parameters : {random_search.best_params_}")
print(f"Best score : {random_search.best_score_}")
print(f"Best Estimator : {random_search.best_estimator_}")

# 24.	Use the fine-tuned model identified during the randomized grid search i.e. the
# best estimator saved in the randomized grid search object to predict the test data
# and note the results it in your written response.
random_search_predictions = random_search.predict(X_test_Ankit)
print_test_scores(random_search_predictions, y_test_Ankit)
plot_heat_map(random_search_predictions, y_test_Ankit)
