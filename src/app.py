from main_ui import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets
import nn
import numpy as np
import sys
import time
import warnings
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15}) #default = 10
class MyApp(Ui_Form):
    def __init__(self, form):
        Ui_Form.__init__(self)
        self.setupUi(form)

        # Redirect output to plainTextEdit
        stream = EmittingStream(text_written=self.__output_written)
        sys.stdout = stream
        sys.stderr = stream

        self.data_file = self.weights_file = ''
        self.train_x = self.train_y = self.test_x = self.test_y = []
        self.model = self.history = None

        # Callbacks
        self.loadDataBtn.clicked.connect(self.__load_data_clicked)
        self.trainBtn.clicked.connect(self.__train_clicked)
        self.saveWeightsBtn.clicked.connect(self.__save_weights_clicked)
        self.loadWeightsBtn.clicked.connect(self.__load_weights_clicked)
        self.predictBtn.clicked.connect(self.__predict_clicked)
        self.tabWidget.currentChanged.connect(self.__tab_changed)

    def __output_written(self, text):
        self.plainTextEdit.moveCursor(QtGui.QTextCursor.End)
        self.plainTextEdit.insertPlainText(text)


# function for data loading
    def __load_data_clicked(self):
        self.data_file = QtWidgets.QFileDialog.getOpenFileName(caption='Load data', filter='Dataset (*.csv)')[0]
        if not self.data_file:
            return
        print('Loaded dataset: {:s}'.format(self.data_file))
        self.train_x, self.train_y, self.test_x, self.test_y = nn.load_and_prepare(self.data_file)

# function for training NN
    def __train_clicked(self):
        if len(self.train_x) == 0:
            print('No data loaded')
            return
        start_time = time.perf_counter()
        self.model, self.history = nn.fit_model(self.train_x, self.train_y,
                                                self.perceptronsSpBox.value(),
                                                self.epochsSpBox.value(),
                                                self.batchSizeSpBox.value())
        mse, mae, mean, std = nn.test_model(self.model, self.test_x, self.test_y)
        self.mseLbl.setText('MSE: {:.4f}'.format(mse))
        self.maeLbl.setText('MAE: {:.4f}'.format(mae))
        self.meanLbl.setText('Mean: {:.4f}'.format(mean))
        self.stdLbl.setText('Stdev: {:.4f}'.format(std))
        self.timeLbl.setText('Training time: {:.2f} sec'.format(time.perf_counter() - start_time))
        if self.drawPlotsCbx.isChecked():
            nn.draw_plots(self.history)

 #function for weights saving
    def __save_weights_clicked(self):
        if not self.model:
            print('No model trained')
            return
        default_file_name = 'weights_{:d}.h5'.format(self.perceptronsSpBox.value())
        self.weights_file = QtWidgets.QFileDialog.getSaveFileName(caption='Save model', filter='Weights (*.h5)',
                                                                  directory=default_file_name)[0]
        if not self.weights_file:
            return
        self.model.save_weights(self.weights_file)
        print('Saved weights: {:s}'.format(self.weights_file))

# func for weights loading
    def __load_weights_clicked(self):
        self.weights_file = QtWidgets.QFileDialog.getOpenFileName(caption='Load model', filter='Weights (*.h5)')[0]
        if not self.weights_file:
            return
        print('Loaded weights: {:s}'.format(self.weights_file))

# func for predicting
    def __predict_clicked(self):
        if len(self.test_x) == 0:
            print('No data loaded')
            return
        if not self.weights_file:
            print('No weights loaded')
            return
        test_x = np.append(self.train_x, self.test_x, axis=0)
        prediction = nn.predict(test_x, self.weights_file, self.perceptronsSpBox.value())

        test_y = np.append(self.train_y, self.test_y, axis=0)

        #sort
        sort_col = test_x[:,0]                #test_y to sort by cs_dose
        prediction = [x for (y, x) in sorted(zip(sort_col, prediction), key=lambda pair: pair[0])]
        test_y = [x for (y, x) in sorted(zip(sort_col, test_y), key=lambda pair: pair[0])]

        plt.plot(prediction)
        plt.plot(test_y, marker=".")

        plt.title('prediction')
        plt.ylabel('dose')
        plt.xlabel('dataset number')
        plt.legend(['Predicted', 'Real'], loc='upper left')
        # plt.legend(['predicted'], loc='upper left')
        plt.show()
        print(prediction)

# func for changing tab with training/testing and loading weights/predicting
    def __tab_changed(self):
        is_tab_1 = (self.tabWidget.currentIndex() == 0)
        self.drawPlotsCbx.setVisible(is_tab_1)
        self.resultsGroupBox.setVisible(is_tab_1)


class EmittingStream(QtCore.QObject):
    text_written = QtCore.pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    my_app = MyApp(Form)
    Form.show()
    sys.exit(app.exec_())
