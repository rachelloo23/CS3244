Decision Tree Analysis: Enhancing Classification Performance
Overview
This branch focuses on evaluating and improving the performance of decision trees in a classification task. Four distinct approaches were implemented to study the impact of feature selection and oversampling techniques on the decision tree's effectiveness. All models perform standard data processing steps, including scaling and cross-validation, to ensure a consistent evaluation framework.
=========================================
Table of Contents
1. Introduction
2. Code Description
3. Approaches
4. Results Summary
5. How to Run
6. Future Directions
Introduction
The decision tree model is known for its simplicity and interpretability. However, it can suffer from overfitting and imbalances in datasets. This branch investigates how the following modifications affect the decision tree's performance:

Feature selection: Selecting the most relevant features to improve generalization.
Oversampling: Addressing class imbalances to enhance performance on underrepresented classes.
Combining feature selection and oversampling.
=========================================
Code Description
This branch contains four Python scripts:

-Plain Decision Tree:
A baseline implementation of a decision tree without any data preprocessing beyond standard scaling and cross-validation.

- Feature Selection on Decision Tree:
Implements a decision tree model after selecting the most relevant features based on correlation analysis, with consistent scaling and cross-validation applied.

- Oversampling on Decision Tree:
Applies oversampling techniques (e.g., SMOTE) to address class imbalances before training the decision tree, followed by scaling and cross-validation.

- Feature Selection and Oversampling on Decision Tree:
Combines feature selection and oversampling.
Applies standard scaling, followed by 10-fold cross-validation.
Additionally computes a confusion matrix to analyze misclassification rates across classes.

=========================================
Approaches
1. Plain Decision Tree
Objective: Establish a baseline for performance comparison.
Process:
Preprocessed the dataset using standard scaling to normalize feature ranges.
Performed 10-fold cross-validation for robust evaluation.
Printed training and testing performance metrics.

2. Feature Selection on Decision Tree
Objective: Improve model generalization by eliminating redundant features.
Process:
Conducted correlation analysis to identify highly correlated features.
Selected top features to reduce noise in the input data.
Applied standard scaling, followed by 10-fold cross-validation.

3. Oversampling on Decision Tree
Objective: Mitigate class imbalances and improve predictions for minority classes.
Process:
Applied SMOTE (Synthetic Minority Oversampling Technique) to oversample underrepresented classes.
Standard scaling was performed after oversampling.
Used 10-fold cross-validation to evaluate the model.

4. Feature Selection and Oversampling on Decision Tree
Objective: Combine the benefits of feature selection and oversampling to maximize performance.
Process:
Performed feature selection and oversampling sequentially.
Applied standard scaling to normalize the dataset.
Conducted 10-fold cross-validation for evaluation.
Calculated the confusion matrix to identify and analyze misclassification rates for each class.
=========================================
Results Summary
Approach	         F1 Score Weighted(Train)   F1 Score Weighted(Test)
Plain Decision Tree	    98.12%	                        82.80%	           
Feature Selection	    97.07%	                        83.21%	     
Oversampling	        100%	               	        83.10%
Combination of Both  	97.28%	               	        84.25%
