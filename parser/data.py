import numpy as np
import pandas as pd
import datetime

from IPython.display import display

class DataProcessor:

    def preprocess_raw_data(self, data, data_columns, data_set_split):
        # Columns present in our data_set
        file_columns = data.columns
        # We will remove all those columns that we won't use 
        # in our model
        diff = set(file_columns) - set(data_columns)
        # We verify if the conlumns we attempt to remove exist in the dataset
        if diff.issubset(file_columns):
            data.drop(diff, axis=1, inplace=True)
        
        data.dropna(axis=0, how='any', inplace=True)
        #We group our dataset hourly
        print("Values head: ")
        display(data.head())
        print("Values tail: ")
        display(data.tail())

        data.index = pd.to_datetime(data.index, unit='s')
        data = data.groupby([pd.Grouper(freq='60Min')]).mean()

        #We will fill any possible missing values with 0
        data.replace(0, np.nan, inplace=True)
        data.fillna(method='ffill', inplace=True)

        print("Values head grouped by hour: ")
        display(data.head())
        print("Values tail grouped by hour: ")
        display(data.tail())
        
        split_index = int(round(data_set_split * len(data))) 
        train_data = data[:split_index]
        test_data = data[split_index:]

        return train_data, test_data
        

    def prepare_data(self, data, data_columns, batch_size, x_window_size, y_window_size, y_col):
        """Performs a data pre-processing in order to split it into different batches containing
        several window time frames. All the created frames will be normalized """
        x_data = []
        y_data = []

        index = 0
        num_rows = len(data)

        while((index+x_window_size+y_window_size) <= num_rows):
            #We split the dataset into the trainig variables and
            #the target variables
            x_window_data = data[index:(index+x_window_size)]
            y_window_data = data[(index+x_window_size):(index+x_window_size+y_window_size)]
            #We will normalize our data for each series
            x_window_data, y_window_data = self.__normalize_series(x_window_data, y_window_data)
            # In case of having multiple values in our y row, we calculate
            # the average of the target value
            y_average = np.average(y_window_data[:, y_col])
            x_data.append(x_window_data)
            y_data.append(y_average)
            index += 1
            
            # index = 100, 200, 300...
            if index % batch_size == 0:
                x_np_arr = np.array(x_data)
                y_np_arr = np.array(y_data)
                x_data = []
                y_data = []
                yield (x_np_arr, y_np_arr)
    
    def __normalize_series(self, data_x, data_y):
        zero_index = data_x.iloc[0]
        y_window = (data_y / zero_index) - 1
        x_window = (data_x / zero_index) - 1
        return (x_window.values, y_window.values)