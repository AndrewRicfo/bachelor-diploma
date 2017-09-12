from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import explained_variance_score, r2_score, median_absolute_error
plt.rcParams.update({'font.size': 15}) #default = 10

def load_and_prepare(file_name):
    dataframe = pandas.read_csv(file_name, engine='python', skipfooter=1)      #reading csv file
    dataset = dataframe.values.astype('float32')    #set data type

    np.random.seed(42)  # fix random seed for reproducibility

    # enforce input randomness
    np.random.shuffle(dataset)

    # remove outliers
    outlier_indices = np.where(dataset[:, 0] > 1.0)[0]
    dataset = np.delete(dataset, outlier_indices, axis=0)
    print('Removed {:d} outliers'.format(len(outlier_indices)))

    dataset, cat_dataset = _prepare_columns(dataset)
    return _split_dataset(dataset, cat_dataset)


def _prepare_columns(dataset):
    # Year_MEASUR, Month_MEASUR to number of months
    new_col = np.array((12 * dataset[:, 3] + dataset[:, 4]) - (12 * 1986 + 4))
    new_col.shape = (len(new_col), 1)

    # PROFESSION, Contamination_zone encoding
    new_prof_columns = _hot_encode(dataset[:, [2]], [1, 2, 3, 4, 5, 6, 7])
    new_zone_columns = _hot_encode(dataset[:, [10]], [2, 3, 4, 24])

    # Soil_type_1/2/3 encoding
    new_soil_columns = _hot_encode(dataset[:, [11, 12, 13]], [1, 2, 3, 4, 29, 31, 32, 35])

    # combine all categorical features
    cat_dataset = new_col  # not categorical, it just doesn't need scaling
    cat_dataset = np.append(cat_dataset, new_prof_columns, axis=1)
    cat_dataset = np.append(cat_dataset, new_zone_columns, axis=1)
    cat_dataset = np.append(cat_dataset, new_soil_columns, axis=1)

    # clean table: profession, year, month, population, edu lvl, zone, soil types
    dataset = np.delete(dataset, [2, 3, 4, 5, 8, 10, 11, 12, 13], axis=1)
    return dataset, cat_dataset


# encoding categorical features
def _hot_encode(features, values):
    values = np.array(values)
    cols = np.zeros((len(features), len(values)))
    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            feature_index = np.where(values == features[i, j])[0]
            if len(feature_index) == 0:
                raise ValueError('Unknown value: {:d}'.format(int(features[i, j])))
            cols[i, feature_index[0]] = 1
    return cols


def _split_dataset(dataset, cat_dataset):
    # split into training and test sets
    train_size = int(len(dataset) * 0.9)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    cat_train, cat_test = cat_dataset[0:train_size, :], cat_dataset[train_size:len(dataset), :]
    print('{:d} train samples, {:d} test samples'.format(len(train), len(test)))

    # extract target values
    train_x = np.delete(train, 0, axis=1)
    train_y = train[:, 0]
    test_x = np.delete(test, 0, axis=1)
    test_y = test[:, 0]

    # scale all numerical features (z-score)
    preprocessing.scale(train_x, copy=False)
    preprocessing.scale(test_x, copy=False)

    # add encoded categorical features
    train_x = np.append(train_x, cat_train, axis=1)
    test_x = np.append(test_x, cat_test, axis=1)

    return train_x, train_y, test_x, test_y


# creating neural network model
def _create_model(input_shape, perceptrons):
    model = Sequential()
    model.add(Dense(perceptrons, activation='tanh', input_shape=input_shape)) #input and hidden layer, tanh activation function
    model.add(Dropout(0.5))   #dropout = 0.5
    model.add(Dense(1))     #output layer
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    return model


# training model
def fit_model(train_x, train_y, perceptrons, epochs, batch_size):
    np.random.seed(42)  # fix random seed for reproducibility

    model = _create_model((train_x.shape[1],), perceptrons)
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.1)
    # list all data in history
    # print(self.history.history.keys())
    return model, history


# testing model
def test_model(model, test_x, test_y):
    # Compare predicted values to actual values
    prediction = model.predict(test_x)
    test_y.shape = (len(test_y), 1)  # for np.hstack()
    np.set_printoptions(suppress=True)  # no scientific notation
    np.set_printoptions(threshold=np.nan)   # print styling
    print('(Test_Y), (Prediction), (Prediction - Test_Y)')
    print(np.hstack((test_y, prediction, prediction-test_y)))       # printing test_y, prediction and their difference
    print("------------------------")
    mear = median_absolute_error(test_y, prediction)  # calculating median absolute error
    print('median absolute error')
    print(mear)

    # Estimate model performance
    mse = np.mean(np.square(prediction - test_y))
    mae = np.mean(np.abs(prediction - test_y))


 # histogram with absolute error distribution
    hist = plt.hist(np.abs(prediction - test_y), bins=75)
    plt.vlines(x=mae, color='red', linewidth=2, ymin=0, ymax=hist[0][np.max(np.where(hist[1] <= mae))], label='MAE')
    plt.vlines(x=mear, color='yellow', ymin=0, linewidth=2,  ymax=hist[0][np.max(np.where(hist[1] <= mear))], label='Median absolute error')
    plt.title('absolute error distribution')
    plt.ylabel('frequency')
    plt.xlabel('absolute error')
    plt.legend()
    plt.show()


    # print('Test scores: {:.4f} MSE, {:.4f} MAE'.format(mse, mae))

    mean = np.mean(prediction)
    std = np.std(prediction)
    return mse, mae, mean, std   # returning testing results


 # predicting dose
def predict(test_x, weights_file, perceptrons):
    model = _create_model((test_x.shape[1],), perceptrons)
    model.load_weights(weights_file)

    prediction = model.predict(test_x)
    return prediction

 # function for plot drawing
def draw_plots(history):
    plt.rcParams["figure.figsize"] = (16, 6)  # default: (8, 6)

    # summarize history for loss/mse
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Median squared error')
    plt.ylabel('MSE value')
    plt.xlabel('Epoch')
    plt.legend(['MSE', 'Validation MSE'], loc='upper right')

    # summarize history for mae
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Median absolute error')
    plt.ylabel('MAE value')
    plt.xlabel('Epoch')
    plt.legend(['MAE', 'Validation MAE'], loc='upper right')

    plt.show()