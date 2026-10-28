# Lab 4 (71616)

## Dataset
UCI Adult Census Income

## Description
Predict whether annual income exceeds $50,000.

## Size
- Rows: 48842
- Columns: 15

## Target
income

## Model
Logistic Regression

## Processing
- Missing value imputation
- Outlier clipping
- Numerical scaling
- Categorical encoding
- Feature engineering

## New features
- capital_net
- hours_per_age

## Evaluation
Accuracy and F1-score on a 30% test set.

## Limitations
The dataset contains historical census information.
Some demographic attributes may introduce fairness concerns.
