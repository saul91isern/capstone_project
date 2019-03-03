import h5py
from parser.data import DataProcessor
from sklearn.externals import joblib

class FileIO:

    def generate_data_files(self, data, conf_data):
        """ Generates a h5py file with all the data ready for the training and testing
        of our model"""
        data_processor = DataProcessor()

        train_data, test_data = data_processor.preprocess_raw_data(
            data, 
            conf_data["util_columns"], 
            conf_data["train_data_split"]
        )

        y_col = list(data.columns).index(conf_data["target_column"])
        #Generate file for the training set
        self.__generate_h5py_file_for_data_set(
            data_processor,
            train_data,            
            conf_data,
            conf_data["train_filename_btc_clean"],
            y_col
        )

        print("Generated data file for the training of the model: ", 
            conf_data["train_filename_btc_clean"]
        )

        #Generate file for the testing set
        self.__generate_h5py_file_for_data_set(
            data_processor,
            test_data,            
            conf_data,
            conf_data["test_filename_btc_clean"],
            y_col    
        )

        print("Generated data file for the testing of the model: ", 
            conf_data["test_filename_btc_clean"]
        )

    def __generate_h5py_file_for_data_set(
        self, 
        data_processor, 
        data_set, 
        conf_data,
        target_filename, 
        y_col
    ):

        """ Creation of the training or testing set of the model """

        util_columns = conf_data["util_columns"]
        batch_size = conf_data["batch_size"]
        x_window_size = conf_data["x_window_size"]
        y_window_size = conf_data["y_window_size"]

        data_gen = data_processor.prepare_data(
            data_set, 
            util_columns, 
            batch_size, 
            x_window_size, 
            y_window_size, 
            y_col
        )
        
        with self.handle_data_file(target_filename, 'w') as hf:
            x_data, y_data = next(data_gen)
            data_set_x = hf.create_dataset("x_data", shape=x_data.shape, maxshape=(None, x_data.shape[1], x_data.shape[2]), chunks=True)
            data_set_y = hf.create_dataset("y_data", shape=y_data.shape, maxshape=(None,), chunks=True)
            
            count_y = y_data.shape[0]
            count_x = x_data.shape[0]

            data_set_x[:] = x_data
            data_set_y[:] = y_data
            
            for x_batch, y_batch in data_gen:
                data_set_x.resize(count_x + x_batch.shape[0], axis=0)
                data_set_y.resize(count_y + y_batch.shape[0], axis=0)

                data_set_x[count_x:] = x_batch
                data_set_y[count_y:] = y_batch

                count_x += x_batch.shape[0]
                count_y += y_batch.shape[0]
            
            self.close_data_file(hf)

    def handle_data_file(self, path, action_type="r"):
        """Performs a reading or writting action to an existing file under a given path 
        and returns it"""
        return h5py.File(path, action_type)
    
    def close_data_file(self, h5pyFile):
        """Close existing data files"""
        try:
            h5pyFile.close()
        except:
            print("File has been already closed")
            pass

    def save_simple_data_file(self, h5pyFile, data, label):
        h5pyFile.create_dataset(label, data=data)
        self.close_data_file(h5pyFile)

    def retrieve_data_from_file(self, h5py_file_data_set, batch_size, start_index=0, end_index=30000):
        """ We read the columns of our generated data file """
        index = start_index
        while index < end_index:
            x_data = h5py_file_data_set["x_data"][index:index+batch_size]
            y_data = h5py_file_data_set["y_data"][index:index+batch_size]
            index += batch_size
            yield(x_data, y_data)

