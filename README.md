# Ensemble-learning
Online lab assignment “Ensemble learning”
Pre-requisite to carrying out the assignment:
1.	Go through and watch all the lectures & lab tutorials of modules 10&11:
2.	We will use the pima-indians-diabetes dataset 
3.	Download the pima-indians-diabetes.csv file and the word description file (metadata).
4.	Study the metadata explained under dataset and attribute information.
5.	Submit one python script for all exercises, as you will be reusing parts of the code.
Assignment due date: end of week # 12
You will have to provide a demonstration video for your solution and upload the video together with the solution on eCentennial through the assignment link. See the video recording instructions at the end of this document. 

Load & check the data:    (5 marks)
1.	Load the data (pima-indians-diabetes.csv)  into a pandas dataframe named df_firstname where first name is you name.
2.	Add the column names i.e. add a header record.
3.	Carryout some initial investigations:
a.	Check the names and types of columns.
b.	Check the missing values.
c.	Check the statistics of the numeric fields (mean, min, max, median, count,..etc.)
d.	Check the categorical values, if any.
e.	In you written response write a paragraph explaining your findings about each column. 
f.	Print out the total number of instances in each class and note into your report and explain your findings in terms of balanced and un-balanced.
Pre-process and prepare the data for machine learning    (10 marks)
4.	Prepare a standard scaler transformer to transform all the numeric values. Name the transformer transformer_firstname.
5.	Split the features from the class.
6.	Split your data into train 70% train and 30% test, use 42 for the seed.  Name the train/test dataframes as follows : X_train_firstname, X_test firstname, y_train firstname, y_test firstname.
7.	Apply(fit, transform the transformer prepared in step 4 to the features i.e. X_train_firstname, X_test firstname. 
(Note: You might need to work on some transformation for the labels in order for the  classifiers to work)
Exercise #1 :Hard voting    (25 marks)
8.	Define 5 classifiers & give them names of your choice, just add at the end _X, where X is the first letter of your last name. Details of classifiers are as follows:
a.	 Logistic Regression set max_iter=1400
b.	 Random Forest Classifier, use the defaults
c.	 Support vector machines use the defaults 
d.	 Decision Tree Classifier set criterion="entropy" & max_depth =42
e.	Extra Trees Classifier, use the defaults 
9.	Define a voting classifier that contains all the above classifiers as estimators, set the voting to hard. 
10.	Fit the training data to the voting classifier and predict the first three instances of test data.
11.	Print out for each classifier (including the voting classifier) and for each instance the predicted and the actual values and note them in your written response.
12.	Note: (One way to achieve the above two steps is a nested loop)	 
13.	In your written response, analyze the results and draw some conclusions. 

Exercise #2: Soft voting            (10 marks)
14.	Repeat steps 8 to 13, except in step 9 set the voting to soft.  
Note: You might need to change some classifiers defaults. 

Exercise #3: Random forests & Extra Trees   (35 marks)
15.	Building on the previous classifiers you defined, create two different pipelines as follows:
a.	Pipeline #1 : Name it pipeline1_firstname. The pipeline should have two steps, the first the transformer you prepared in step #4 and the second the Extra Trees Classifier you prepared in step 8.e.
b.	Pipeline #2 : Name it pipeline1_firstname. The pipeline should have two steps, the first the transformer you prepared in step #4 and the second the Decision Tree Classifier you prepared in step 8.d.

16.	Fit the original data to both pipelines.
17.	Carry out a 10 fold cross validation for both pipelines set shuffling to true and random_state to 42.
18.	Printout the mean score evaluation for both pipelines, note the final results in your written response and advise which pipeline performed better.
19.	Predict the test using both pipelines and printout the confusion matrix, precision, recall and accuracy scores and record the results in your written response. (Use a loop)
20.	Compare the results and compare them against each other and against the accuracy scores you recorded during step #18, write some conclusions.
Exercise #4: Extra Trees and Grid search       (15 marks)
21.	Carryout a randomized grid search on the first Pipeline you defined in step #15. The randomized grid search should investigate the use of different values for the following two parameters:
a.	The number of trees in the forest. Use a distribution between the values of 10 trees and 3000 trees with a step of 20.
b.	The maximum depth of the tree.  Use a distribution between the values of 1 max depth and 1000 max_depth with a step of 2.
Choose appropriate names for both your grid search parameter objects that end with_XX, where XX is the last two digits of your student id.
22.	Fit your training data to the randomized gird search object.
23.	Print out the best parameters and accuracy score for randomized grid search and record them in your written response and comment on these results. For example, compare the accuracy score to the one you obtained in step # 17 and compare the parameters suggested by the randomized grid search to the ones you used in the classifier setup i.e. step #8. list any conclusions you noticed.
24.	Use the fine-tuned model identified during the randomized grid search i.e the best estimator saved in the randomized grid search object to predict the test data and note the results it in your written response.
25.	Printout the precision, re_call and accuracy. Compare the accuracy score with earlier readings you generated during steps 23. Are the better or worse explain why.
To learn how to use randomized grid search check the following reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
