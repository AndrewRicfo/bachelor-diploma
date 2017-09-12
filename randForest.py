from sklearn.ensemble import RandomForestRegressor
import numpy as np
import nn
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error


train_x, train_y, test_x, test_y = nn.load_and_prepare("D:\studying\dplm\data_nc_full_nice.csv")


clf = RandomForestRegressor()
clf.fit(train_x, train_y)
clf.score(train_x, train_y)

predicted = clf.predict(test_x)

mse = np.mean(np.square(predicted - test_y))
mae = np.mean(np.abs(predicted - test_y))
mear = median_absolute_error(test_y, predicted)

print("mse, mae, mear")
print(mse, mae, mear)

plt.plot(predicted)
plt.plot(test_y)
plt.show()