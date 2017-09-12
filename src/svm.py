#Import Library
from sklearn import svm
import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import median_absolute_error

train_x, train_y, test_x, test_y = nn.load_and_prepare("D:\studying\dplm\data_nc_full_nice.csv")

model = svm.SVR(C=1.0, epsilon=0.2)

# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(train_x, train_y)
model.score(train_x, train_y)

# Predict Output
predicted = model.predict(test_x)

mse = np.mean(np.square(predicted - test_y))
mae = np.mean(np.abs(predicted - test_y))
mear = median_absolute_error(test_y, predicted)

print("mse, mae, mear")
print(mse, mae, mear)

plt.plot(predicted)
plt.plot(test_y)
plt.show()
