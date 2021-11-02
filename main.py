import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# LinearRegression(fit_intercept = True, normalize = False, copy_X = True, n_jobs = 1) 

dataset = pd.read_csv("data/dummy_with_vblank.csv")
X = dataset.iloc[:, 0].values.reshape(-1,1)
y = dataset.iloc[:,1].values
# split into train and test dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# print("X_train: ", X_train)
print("X_test: ", X_test)
# print("Y_train: ", y_train)
print("Y_test: ", y_test)

# train and create the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# calculate the intercept and coefficient.
intercept = regressor.intercept_
coefficient = regressor.coef_
print('Interception : ', intercept)
print('Coeficient : ', coefficient)

# calculate the score
score = regressor.score(X_test, y_test)
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')

# take test_data in x to predict y
y_pred = regressor.predict(X_test)
print('Predict : ', y_pred)


# plot the results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Dummy Line vs Vblank time (trainning set)')
plt.xlabel("Number of Dummy Lines")
plt.ylabel("Vblank Time")
plt.show()

# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Dummy Line vs Vblank time (test data set)')
# plt.xlabel("Number of Dummy Lines")
# plt.ylabel("Vblank Time")
# plt.show()

print("So the equation is noted as:")
print("Vblank(line)=",round(coefficient[0],5),'*line+',round(intercept,5))