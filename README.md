# Sentiment Analysis with Naive Bayes, Logistic Regression, and SVM

## Overview
This assignment aims to classify movie reviews as either positive or negative using three machine learning models: Naive Bayes, Logistic Regression, and Support Vector Machines (SVM). The dataset consists of movie reviews that are preprocessed, vectorized using TF-IDF, and then used for training and evaluation.

## Models Used
1. **Naive Bayes**: A simple yet effective probabilistic classifier.
2. **Logistic Regression**: A linear model for binary classification tasks.
3. **Support Vector Classifier (SVC)**: A robust model that finds the optimal decision boundary between classes.

## Steps

### 1. Data Preprocessing
- Loaded text data from two files (`rt-polarity.pos` for positive and `rt-polarity.neg` for negative reviews).
- Cleaned the text by removing stopwords and non-alphabetical characters.

### 2. Feature Extraction
- Used `TfidfVectorizer` to transform the preprocessed text into numerical vectors based on Term Frequency-Inverse Document Frequency (TF-IDF).

### 3. Model Training
- Trained three models:
  - **Naive Bayes** (`MultinomialNB`)
  - **Logistic Regression**
  - **SVM (Support Vector Classifier)**
  
### 4. Model Evaluation
- Evaluated the models on the test data using the following metrics:
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - Accuracy

## Results

### Naive Bayes:
- **Accuracy**: 77.62%
- **Confusion Matrix**:
  - True Positives (TP): 641
  - True Negatives (TN): 649
  - False Positives (FP): 182
  - False Negatives (FN): 190

## Conclusion
- Naive Bayes achieved an accuracy of **77.62%**, with balanced precision and recall.
- Logistic Regression and SVM were also trained, and results were compared using the same evaluation metrics.
