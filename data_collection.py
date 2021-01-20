import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

'''This Class will preprocess the data an apply min max normalisation'''


class MyCollection:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.scaler = MinMaxScaler()
        self.min_values = None
        self.max_values = None

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath, header=None)

    def preprocessing(self):
        self.scaler.fit(self.df)
        self.min_values = self.scaler.data_min_
        self.max_values = self.scaler.data_max_
        transformed_df = self.scaler.transform(self.df)
        x = transformed_df[:, :2]
        y = transformed_df[:, 2:]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=30)
        # print("Min: ", self.min_values)
        # print("Max: ", self.max_values)
        return self.x_train, self.x_test, self.y_train, self.y_test
