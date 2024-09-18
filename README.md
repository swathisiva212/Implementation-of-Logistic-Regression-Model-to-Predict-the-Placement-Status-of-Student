# Implementation of Logistic Regression Model to Predict the Placement Status of Student

## Aim:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Get the data and use label encoder to change all the values to numeric.
2. Drop the unwanted values,Check for NULL values, Duplicate values.
3. Classify the training data and the test data.
4. Calculate the accuracy score, confusion matrix and classification report. 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SWATHI.S
RegisterNumber: 212223040219.
```

```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Predicted values : ")
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy : ")
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification Report : ")
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
#### Dataset
![Screenshot 2024-03-22 104100](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/65acc46e-6329-4d2f-81ab-172fe5748260)

#### Transformed Data
![Screenshot 2024-03-22 104335](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/5e454aac-6d5e-4112-9bf8-6edfdad70d4f)

#### Null values
![Screenshot 2024-03-22 104228](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/6a9d8e2b-332a-4e77-85dd-65c35b1629e7)

#### X values
![Screenshot 2024-03-22 104439](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/fb0f3e51-a0df-4ddc-94c3-9adecf7822af)

#### Y values
![Screenshot 2024-03-22 104523](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/394b819e-7e59-4c4b-bf1d-42bccc97320c)


![Screenshot 2024-03-22 104603](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/8c07787b-5f93-4d01-b982-866ecce597b9)


![Screenshot 2024-03-22 104646](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/676b7c68-d539-4063-b453-62a71a5ebe78)


![Screenshot 2024-03-22 105013](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/2f76995d-71c2-45a7-aa5f-7e3e5d49074d)

#### Classification Report
![Screenshot 2024-03-22 105104](https://github.com/Jenishajustin/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405070/6a471ef9-2451-4a4a-8d6c-854e6dd66c89)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.


