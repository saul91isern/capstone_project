from keras.layers import Activation, BatchNormalization, Dense, Dropout, Embedding, LSTM, GRU
from keras.models import load_model, Sequential

class NotImplementedModel ( Exception ): pass
    
class ModelBuilder:

    def build_gru_model(self, layers, conf_model, conf_data):
        
        model = Sequential()

        model.add(GRU(
            units=layers[1],
            activation=conf_model["activation_function"],
            return_sequences=True,
            dropout=0.20,
            input_shape=(conf_data["x_window_size"], layers[0])
            )
        )

        model.add(GRU(
            units=layers[2],
            activation=conf_model["activation_function"],
            return_sequences=True,
            dropout=0.20
            )
        )

        model.add(GRU(
            units=layers[3],
            activation=conf_model["activation_function"],
            return_sequences=False,
            dropout=0.25
            )
        )

        model.add(Dense(layers[4], activation=conf_model["activation_function"]))

        model.summary()

        model.compile(
            optimizer=conf_model["optimizer_function"],
            loss=conf_model["loss_function"]
        )

        return model
    
    def train_model(
        self, 
        model, 
        train_data, 
        validation_data, 
        conf
    ):
        
        if not model:
            raise NotImplementedModel("Not Implemented model")

        history = model.fit_generator(
            train_data, 
            steps_per_epoch=conf["steps_per_epoch"], 
            epochs=conf["epochs"],
            validation_data=validation_data,
            validation_steps=conf["validation_steps"],
            shuffle=False
        )

        # Once finished the training the model it will be saved and we will delete
        # the class attribute from memory.
        model.save(conf["filename_generated_model"])
        del model
        return history
    
    def generate_predictions(self, conf, test_data):
        model = load_model(conf["filename_generated_model"])
        return model.predict_generator(self.__generator(test_data), steps=conf["test_steps"])

    def __generator(self, test_data):
        for x, _ in test_data:
            yield x


