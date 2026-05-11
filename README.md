# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Algorithm
2. Load employee data and split it into training and testing sets.
3. Train a Decision Tree classifier using entropy as the split criterion.
4. Evaluate the model using accuracy, confusion matrix, and classification report.
5. Use the trained model to predict whether a new employee will stay or leave.
## Program:
```/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: TEJA D P
RegisterNumber:212225040464
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier(max_iter=1000, tol=1e-3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

## Output:
<img width="565" height="301" alt="image" src="https://github.com/user-attachments/assets/45ef0048-6917-400f-af62-84dac923367d" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
