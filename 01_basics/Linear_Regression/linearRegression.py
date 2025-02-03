# Learning linear regression

# Goal  predict student grade based on past grades
# Learning linear regression

# Goal  predict student grade based on past grades

import pandas as pd
import numpy as np

data = pd.read_esv("student-mat.csv",sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# label->what we are predicting

predict = "G3"

# Training data data frame without predict(G3)
x = np.array(data.drop([predict], 1))

y = np.array(data[predict])

# x_train a postion of x(training data ) & y_train is a portion of y data(predicted data) and x & test
# is determined by test_size which is 10% of the training data
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)
