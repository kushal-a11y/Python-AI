# Learning linear regression

# Goal  predict student grade based on past grades
# Learning linear regression

# Goal  predict student grade based on past grades

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import math
from matplotlib import style

data = pd.read_csv("student-mat.csv",sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# label->what we are predicting

predict = "G3"

# Training data data frame without predict(G3)
x = np.array(data.drop([predict], axis = 1))

y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

# best = 0
# for __ in range(500):

# # x_train a postion of x(training data ) & y_train is a portion of y data(predicted data) and x & test
# # is determined by test_size which is 10% of the training data
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

# #Building the linear regression model, fitting datat into in that model and measure score against data
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train,y_train)
#     accuracy = linear.score(x_test,y_test)

#     if accuracy > best:
#         best = accuracy
#         with open("studentModel.pickle","wb") as f:
#             pickle.dump(linear, f)
    

pickle_in = open("studentModel.pickle","rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test,y_test)

predictions = linear.predict(x_test)

print(f"accuracy: {math.ceil(acc * 100)}%")

for i in range(len(predictions)):
    print(f"My prediction : {math.floor(predictions[i])} || Test data: {x_test[i]} || Actual result: {y_test[i]}")
p = "G1"
style.use("ggplot")
plt.ion()
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("G3")
plt.draw()  # Updates the plot
plt.pause(2)  # Keeps the plot open for 2 seconds
plt.ioff()  # Turns off interactive mode
plt.show() 
plt.show()