# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the numpy module as np and in next step import mathplot.pyplot as plt module and import the panda module as pd.
2. Load the dataset by using pd.readscv() to read csv by using panda module.
3. Import sklearn.metrics to import mean_absolute_error and mean_squared_error.
4. Show the dataframe using tail or head.
5. Create variable 'X' and give the input using indexlocation(df.iloc[:,:-1]).
6. Create variable 'Y' and give the input using indexlocation(df.iloc[:,1]).
7. Import sklearn.model_selection from that import train_test_split.
8. Create new variable as X_train,X_test,Y_train,Y_test and give input as train_test_split and give this inside parameter (X,Y,test_size=1/3,random_state=0).
9. Import sklearn.linear_model from that import LinearRegression.
10. Use regression variable to use LinearRegression module.
11. Use fit for X_train and Y_train variable as training sets for that model(dataframe).
12. create varaible y_pred and give input as regression.predict(X_test) to store predicted value for Y.
13. show the predicted value(y_pred) and  Y_test to show the actual value.
14. Use mathplot.pyplot to show graph between X_train and Y_train using scatterplot for training set.
15. Use mathplot.pyplot to show graph between X_train and X_train of predicted value using lineplot for training set.
16. Use mathplot.pyplot to show graph between X_test and Y_test using scatterplot for test set.
17. Use mathplot.pyplot to show graph between X_test and X_test predicted value using lineplot for test set.
18. Use Mean square error and mean absolute error and Root mean square error to calculate error rate between Y_test and y_pred.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VESHWANTH.
RegisterNumber: 2l2224230300
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:\\Users\\admin\\Desktop\\DS LAB FILES\\machine learning\\DATASET-20250226\\student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,Y_train)
y_pred=regression.predict(X_test)
y_pred

plt.scatter(X_train,Y_train,color='red',marker='o',label='scatter')
plt.plot(X_train,regression.predict(X_train),color='Green',marker='d',label='lineplot')
plt.title("Hours vs Scores (Training sets)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()

plt.scatter(X_test,Y_test,color='orange',marker='v',label='scatter')
plt.plot(X_test,regression.predict(X_test),color='blue',marker='D',label='lineplot')
plt.title("Hours vs Scores (Test sets)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()

mse=mean_squared_error(Y_test,y_pred)
print("MSE = ",mse)

mae=mean_absolute_error(Y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

*/
```

## Output:
![ex 2 1](https://github.com/user-attachments/assets/2a2917a2-7a2f-495e-a04e-165f4dbd0ec3)
![ex 2 2](https://github.com/user-attachments/assets/1b2cc70e-34af-4e67-b8eb-16e24f7f68e1)
![2 3](https://github.com/user-attachments/assets/82fe048c-f91b-4d6a-9231-9dc4ee34614f)
![2 4](https://github.com/user-attachments/assets/02dc1588-b323-4307-a391-43689409f619)
![2 5](https://github.com/user-attachments/assets/19270896-018d-406d-9638-0a482a5f1c71)
![2 6](https://github.com/user-attachments/assets/1cacd83e-4025-44ce-a39b-736b85d50085)
![2 7](https://github.com/user-attachments/assets/044b08ce-ea2b-48b4-99a1-8ed1a3a72a31)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
