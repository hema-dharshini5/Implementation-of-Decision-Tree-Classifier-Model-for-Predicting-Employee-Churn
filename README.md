# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.

## Program:
```

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Hema dharshini N
RegisterNumber:  212223220034
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee (1).csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![ml61](https://github.com/hema-dharshini5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147117728/8ace7d7b-d2d3-4010-9871-4ca49ef4d613)
![ml 6 2](https://github.com/hema-dharshini5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147117728/c4abdfdd-eea3-4a74-91be-670a9c42d8ee)

![ml 6 3](https://github.com/hema-dharshini5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147117728/4bd8d065-4850-4cd7-a8d6-2f7b74a28748)

![ml 6 4](https://github.com/hema-dharshini5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147117728/e3ab19cc-0c82-4193-ba94-07693ac0d7ac)


![ml 6 5](https://github.com/hema-dharshini5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147117728/a27fc94d-b094-48c7-824f-2b8a438628c3)

![ml 6 6](https://github.com/hema-dharshini5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147117728/3450360f-010e-4daa-8bad-74d06e1e7095)

![322720580-5c88de7e-879f-4c53-b2d3-780a8ac6a7d5](https://github.com/hema-dharshini5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147117728/6bf1b619-580a-446f-ad68-7e40cb0455ef)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
