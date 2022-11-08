# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## 1.Import the standard libraries.
## 2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
## 3.Import LabelEncoder and encode the dataset.
## 4.Import LogisticRegression from sklearn and apply the model on the dataset.
## 5.Predict the values of array.
## 6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
## 7.Apply new unknown values. 
 
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VARSHA AJITH
RegisterNumber:  212221230118
*/

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![o1](https://user-images.githubusercontent.com/94222288/200585900-c462b66b-06ef-4e3c-a808-4848a0b6eb04.png)
![o2](https://user-images.githubusercontent.com/94222288/200585913-40faf827-a6ff-40a8-8cf8-69d1d35f4cd9.png)
![o3](https://u![o5](https://user-images.githubusercontent.com/94222288/200587345-d89a1186-6567-4d00-ad09-c439a2eca1ba.png)

![o4](htt![o5](https://user-images.githubusercontent.com/94222288/200587266-ce4c3e6c-0aca-4608-837c-2c4926e6ae7a.png)

![o5](https://user-images.githubusercontent.com/94222288/200587422-79454a19-9e0c-4078-a1aa-07b6ea860331.png)


![o6](https://user-images.githubusercontent.com/94222288/200587473-80e6d299-4c66-4a81-a6c1-54d49f956b7e.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
