# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Dinesh V
RegisterNumber:  212224040076
*/
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])

```

## Output:

<img width="1590" height="253" alt="500351830-a4ca95e4-b474-44e5-9399-294c1a02cd18" src="https://github.com/user-attachments/assets/ea13be87-0e3b-4eab-88d9-f919d848e1e3" />



<img width="505" height="420" alt="500347538-7c9f6147-20cf-4e35-a364-88a70393082c" src="https://github.com/user-attachments/assets/17c06df8-0be4-47e9-abed-9c902a809a03" />



<img width="1385" height="368" alt="500347973-a3304000-385e-4f7b-9881-5f836a650314" src="https://github.com/user-attachments/assets/e4cb422c-2351-4ffb-9a04-7c755b409bf1" />

<img width="249" height="534" alt="500347618-b115552d-8781-4d03-a74a-4cc4c67d5129" src="https://github.com/user-attachments/assets/a75df6f7-7d3a-4f96-8037-d074fa80e9a1" />


<img width="189" height="76" alt="500347700-bfdb2ba2-08b1-4510-86b4-ebc7f393a391" src="https://github.com/user-attachments/assets/edb94b05-ec33-41bf-b846-d2d4c030bd3f" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
