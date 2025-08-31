# Medical Diagnosis Classifier: Handling Imbalanced Data

This project explores **machine learning approaches for medical diagnosis** using a dataset of appendicitis cases. The primary challenge tackled here is **class imbalance**, where the "appendicitis" class is overrepresented compared to the "no appendicitis" class.  

The notebook implements and compares multiple strategies to mitigate imbalance, improve recall on the minority class, and ensure clinically meaningful performance.  

---

## 🔑 Key Features
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

---

## Results Overview
- **Baseline models** showed bias toward the majority class ("appendicitis").  
- **Resampling and class-weighting strategies** improved recall and F1 for the minority class ("no appendicitis").  
- **GridSearch optimization** achieved the best trade-off between overall performance and minority class sensitivity.  
- Visualizations demonstrate the incremental improvements in F1 scores for each strategy.  

---

##  Tech Stack
- **Python** (pandas, scikit-learn, matplotlib, imbalanced-learn)  
- **Machine Learning** (classification, cross-validation, model optimization)  
- **Data Handling** (preprocessing, resampling, scaling)  

---

## Project Structure
- `appendicitis.csv` — dataset  
- `imbalanced-appendicitis-diagnosis.ipynb` — main implementation
- `requirements.txt` — main implementation
- `README.md` — project overview

---

## Learning Outcomes

This project demonstrates:

- How to design fair evaluation strategies for imbalanced medical datasets.  
- The importance of recall and F1-score in healthcare ML applications.  
- How resampling and hyperparameter tuning impact model performance.  
