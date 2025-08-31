# Student Performance Prediction: Handling Imbalanced Data

This project explores **machine learning approaches for predicting student performance**, with a particular focus on handling **class imbalance**.  
The dataset contains student-related features, and the goal is to accurately predict whether a student will fall into a certain performance category.  

The notebook implements and compares multiple strategies to mitigate imbalance, improve recall on the minority class, and ensure fair and reliable performance.  


## Key Features
- **Models Implemented:**
  - k-Nearest Neighbors (kNN)
  - Decision Trees  
- **Imbalance Handling Techniques:**
  - Class weighting  
  - SMOTE (Synthetic Minority Oversampling Technique)  
  - Random oversampling & NearMiss undersampling  
  - Hyperparameter tuning via GridSearchCV  
- **Evaluation Metrics:**  
  - F1-score (macro & per class)  
  - Recall (per class)  
  - Stratified k-fold cross-validation  


## Results Overview
- **Baseline models** showed bias toward the majority class.  
- **Resampling and class-weighting strategies** improved recall and F1 for the minority class.  
- **GridSearch optimization** achieved the best trade-off between overall performance and minority class sensitivity.  
- Visualizations demonstrate the incremental improvements in F1 scores for each strategy.  


## Tech Stack
- **Python** (pandas, scikit-learn, matplotlib, imbalanced-learn)  
- **Machine Learning** (classification, cross-validation, model optimization)  
- **Data Handling** (preprocessing, resampling, scaling)  


## Project Structure
- `student-performance.csv` — dataset  
- `student-performance-prediction.ipynb` — main implementation  
- `requirements.txt` — dependencies  
- `README.md` — project overview  



## Learning Outcomes

This project demonstrates:

- How to design fair evaluation strategies for imbalanced datasets.  
- The importance of recall and F1-score in evaluating model performance.  
- How resampling and hyperparameter tuning impact classification results.  

