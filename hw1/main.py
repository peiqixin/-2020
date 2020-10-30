# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_name):
    csv_file = pd.read_csv(file_name, encoding='utf-8')
    csv_file = csv_file[csv_file['observation'] == 'PM2.5']
    csv_file = csv_file.drop(['Date', 'stations', 'observation'], axis=1)
    x1 = []
    y1 = []
    for i in range(15):
        x = csv_file.iloc[:, i: i + 9]
        y = csv_file.iloc[:, i + 9]
        x.columns = np.array(range(9))
        y.columns = np.array(range(9))
        x1.append(x)
        y1.append(y)
    # concat: merge DataFrame
    x1, y1 = pd.concat(x1), pd.concat(y1)
    x1 = np.array(x1, dtype=D_TYPE)
    test_csv = pd.read_csv('Dataset\\test.csv', encoding='utf-8')
    test_csv = test_csv[test_csv['AMB_TEMP'] == 'PM2.5']
    x2 = test_csv.iloc[:, 2:]
    x2 = np.array(x2, dtype=D_TYPE)
    # TODO** scandalize data
    ss = StandardScaler()
    ss.fit(x1)
    x1 = ss.transform(x1)
    ss.fit(x2)
    x2 = ss.transform(x2)

    return x1, np.array(y1, dtype=D_TYPE), x2


class LinearRegression:

    def __init__(self):
        self.theta = None

    def loss_func(self, x, y):
        return np.sum(0.5 * (x.dot(self.theta) - y) ** 2) / len(y)

    @staticmethod
    def r2_score(y_hat, y):
        # Mean Squared Error 均方误差
        mse = np.sum((y_hat - y) ** 2) / len(y_hat)
        # np.var() compute variance
        return 1 - mse / np.var(y_hat)

    def training(self, x, y, epoch=10000, learning_rate=0.01, epsilon=1e-5):
        # add bias
        print(x.shape)
        # x = pd.concat([np.ones((len(x), 1)), x], axis=1)
        x = np.hstack([np.ones((len(x), 1)), x])
        self.theta = np.zeros(x.shape[1])

        prev_loss = 1e9
        # gradient descent
        for i in range(epoch):
            # dJ/dTheta
            cur_loss = self.loss_func(x, y)
            print(cur_loss)
            if abs(cur_loss - prev_loss) < epsilon:
                break
            prev_loss = cur_loss
            predict_y = x.dot(self.theta)
            d_theta = x.T.dot(predict_y - y) / len(y)
            self.theta = self.theta - learning_rate * d_theta

    def predict(self, x):
        x = np.hstack([np.ones((len(x), 1)), x])
        return x.dot(self.theta)


if __name__ == '__main__':
    D_TYPE = 'float'
    train_x, train_y, test_x = load_data('Dataset\\train.csv')
    # print(train_x, train_y)
    print(type(train_x))

    model = LinearRegression()
    model.training(train_x, train_y)
    result = model.predict(test_x)

    # save result to csv
    sample = pd.read_csv('Dataset\\sampleSubmission.csv', engine='python', encoding='utf-8')
    sample['value'] = result
    sample.to_csv('Dataset\\result.csv')
