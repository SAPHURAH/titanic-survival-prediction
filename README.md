# titanic-survival-prediction
This project utilizes machine learning techniques to predict survival on the Titanic based on passenger data. It includes data preprocessing, exploratory data analysis, feature engineering, and model training using a Random Forest Classifier. The goal is to analyze how factors like age, sex, and passenger class influenced survival rates.

Here’s a simplified documentation for your Titanic machine learning project, detailing the entire process you went through, from setup to completion. This document can be saved as a README.md file in your project repository.

---

# Titanic Survival Prediction Project

## Introduction
This project aims to predict whether passengers survived the Titanic disaster using machine learning techniques. We utilized the Titanic dataset, which contains various details about the passengers, such as age, gender, ticket class, and more.

## Project Overview
The project involves several steps, including data exploration, preprocessing, model training, and evaluation. The main goal is to build a model that can accurately predict survival based on the given features.

## Steps Taken

### 1. Setup and Libraries
To start the project, I set up my Python environment and installed the necessary libraries. The main libraries used in this project are:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.
- `scikit-learn`: For machine learning algorithms.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### 2. Data Loading
I loaded the Titanic dataset, which was provided in CSV format. The dataset contains information about 887 passengers.

```python
data = pd.read_csv('titanic.csv')
```

### 3. Data Exploration
I explored the dataset to understand its structure and the features available. I used functions like `data.head()` and `data.info()` to check the first few rows and the data types.

#### Visualization
I created visualizations to understand the survival distribution and how various factors, like passenger class and gender, affected survival rates.

```python
# Survival Distribution
sns.countplot(x='Survived', data=data)
plt.title('Survival Distribution on the Titanic')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Survival Rates by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival Rates by Passenger Class')
plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
```

### 4. Data Preprocessing
I cleaned the data by handling missing values and encoding categorical variables.

- **Handling Missing Values**: I filled in missing ages with the median age and filled missing fares with the median fare.

```python
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
```

- **Encoding Categorical Variables**: I converted the 'Sex' column into numeric values (0 for male, 1 for female).

```python
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
```

### 5. Feature Selection
I selected the features that would be used for training the model. I chose `Pclass`, `Sex`, `Age`, `SibSp` (number of siblings/spouses aboard), and `Fare`.

```python
features = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]
target = data['Survived']
```

### 6. Splitting the Data
I split the data into training and testing sets to evaluate the model's performance. I used 80% of the data for training and 20% for testing.

```python
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
```

### 7. Model Training
I chose the Random Forest Classifier for training because of its effectiveness in classification tasks. After training the model, I evaluated its accuracy.

```python
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
train_accuracy = model.score(X_train, y_train)
```

### 8. Model Evaluation
I evaluated the model using the testing data and calculated its accuracy. I also created a confusion matrix and a classification report to understand the model's performance.

```python
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
```

### 9. Results
The final results of the project showed:

- **Training Accuracy**: 97.89%
- **Testing Accuracy**: 80.45%
- **Confusion Matrix**:
  ```
  [[90 15]
   [20 54]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

             0       0.82      0.86      0.84       105
             1       0.78      0.73      0.76        74

      accuracy                           0.80       179
     macro avg       0.80      0.79      0.80       179
  ```

### 10. Challenges Faced
Throughout the project, I encountered several challenges:
- **Handling Missing Values**: I initially struggled with how to address missing values in the dataset.
- **SettingWithCopyWarning**: I received warnings when modifying the DataFrame. I learned to use `.loc[]` to avoid these warnings.
- **Model Tuning**: I experimented with different algorithms and parameters to improve accuracy.

### Conclusion
This project enhanced my understanding of data preprocessing, machine learning, and model evaluation. I learned valuable skills in handling real-world datasets and implementing machine learning models.

## Future Improvements
In the future, I plan to:
- Experiment with other algorithms, like logistic regression or support vector machines.
- Use more advanced techniques like hyperparameter tuning to improve model performance.
- Explore other datasets to practice and enhance my skills further.
