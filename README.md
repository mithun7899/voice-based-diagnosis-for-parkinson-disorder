
# 🧠 Parkinson's Disease Prediction Web App 

This project aims to develop a machine learning model with the capability to make accurate predictions regarding the presence of Parkinson's disease in individuals based on the analysis of their voice recordings. The model is deployed in a simple flask web app where you can enter the csv inorder to predict wether the person has PD or not.

## Problem 🎯
Parkinson's disease is a progressive neurodegenerative disorder primarily recognized for its impact on motor functions, giving rise to symptoms such as tremors, muscular rigidity, and difficulties in maintaining coordinated movements. By utilizing advanced computational techniques, this project seeks to establish a reliable tool for early detection and potential monitoring of Parkinson's disease through non-invasive voice analysis.

## Steps 📊
1. **Explore the Dataset**: Reveal the underlying trends, occurrences, and connections present in the data by identifying patterns, distributions, and relationships.
2. **Preprocessing Steps**: Remove irrelevant features, address missing values, treat outliers, encode categorical variables.
3. **Conduct Extensive Exploratory Data Analysis (EDA)**: Dive deep into bivariate relationships against the target.
4. **Model Building**: Establish pipelines for models that require scaling, implement classification models including KNN, SVM, Decision Trees and Random Forest.
5. **Evaluate and Compare Model Performance**: Utilize precision, recall,  F1-score, accuracy and cross validation score to gauge models' effectiveness.

## Requirements 📚
```bash
pip install -r requirements.txt
```

## Models 📝
Models demonstrate high accuracy and F1-scores, indicating strong performance. However, there are some differences to consider:
- Random Forest & XGBoost achieves higher metrics (BUT we might have risks of overfitting).
- K Neighbors achieves good metric less than the other two yet less risk of overfitting.
Given the performance of both models, either could be a suitable choice.


## Output of the project

![Screenshot 2025-04-23 052406](https://github.com/user-attachments/assets/61b454b0-ac69-4c72-90cf-643453729c84)
![Screenshot 2025-04-23 052436](https://github.com/user-attachments/assets/0d58552f-6b27-47d9-9a5c-60e684e72b88)
![Screenshot 2025-04-23 052458](https://github.com/user-attachments/assets/b5cc564b-e90d-43ca-8bef-2afc561c391c)
![Screenshot 2025-04-23 052522](https://github.com/user-attachments/assets/a791ce4c-c1d5-469b-b2b8-f271782294d8)
![Screenshot 2025-04-23 052543](https://github.com/user-attachments/assets/d869d5eb-be67-4f1e-b9af-95207e561770)
![Screenshot 2025-04-23 052554](https://github.com/user-attachments/assets/36d666ff-a77f-47a7-8f62-01e44eeaa721)
![Screenshot 2025-04-23 052605](https://github.com/user-attachments/assets/02e082d6-df73-47a8-ae13-dafb9232e6a5)
![Screenshot 2025-04-21 151821](https://github.com/user-attachments/assets/10d1c209-0928-4f3c-b2ec-eb2104dd0a7b)
![Screenshot 2025-04-21 151921](https://github.com/user-attachments/assets/8583468e-7e8c-4046-8adc-b8c4d5ab9553)




