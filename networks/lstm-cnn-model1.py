from utils import *
from datagen.SpectraGenerator import SpectraGenerator
from networks.SpectraPreprocessor import SpectraPreprocessor
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, LSTM, TimeDistributed, MaxPooling2D, BatchNormalization, Dropout, \
    Conv1D, MaxPooling1D, Bidirectional, CuDNNGRU, Reshape, Concatenate, concatenate, Input, CuDNNLSTM
from keras.optimizers import SGD, Adam
from networks.attention import Attention


def main():
    print(f"Running model {use_version}")
    print(f"Creating Preprocessor for set {dataset_name}")
    spectra_preprocessor = SpectraPreprocessor(dataset_name=dataset_name, use_generator=True)
    #spectra_preprocessor = SpectraPreprocessor(dataset_name=dataset_name, use_generator=False)
    print("Splitting dataset")
    X_test, y_test = spectra_preprocessor.transform_test(encoded=True)
    #X_train, y_train, X_test, y_test = spectra_preprocessor.transform(encoded=True)

    lstm_model = build_model(use_version)

    baseline_model_compile_dict = {'optimizer': optimizer,
                                   'loss': loss,
                                   'metrics':['accuracy', 'mae', 'mse']}

    baseline_model = BaseModel(lstm_model)
    #baseline_model.fit(X_train, y_train, X_test, y_test, batch_size=batch_size, epochs=n_epochs, compile_dict=baseline_model_compile_dict)
    baseline_model.fit_generator(spectra_preprocessor, spectra_preprocessor.datagen_config["num_instances"], X_test, y_test, batch_size=batch_size, epochs=n_epochs, compile_dict=baseline_model_compile_dict, encoded=True)

    save_model(model_name=save_name, model= baseline_model, train_path=train_path, test_path=test_path)
    baseline_model.keras_model.save("%s.h5" % save_name)

def build_model(version=0):

    if version == 0:
        model = Sequential()

        # define CNN model
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(1001, 10, 1)))
        model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1), padding='valid'))
        model.add(Flatten())
        #model.add(LSTM(10, return_sequences=False))

        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 1:
        cnn = Sequential()
        cnn.add(Conv1D(64, kernel_size=5, input_shape=(10,1), activation='relu'))
        cnn.add(MaxPooling1D(pool_size=2))
        cnn.add(Flatten())

        model = Sequential()
        model.add(TimeDistributed(cnn, input_shape=(1001, 10)))
        model.add(LSTM(10))
        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 2:
        cnn = Sequential()
        cnn.add(Conv2D(64, kernel_size=3, activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2,1), strides=(2,1), padding='valid'))
        cnn.add(Flatten())

        model = Sequential()
        model.add(TimeDistributed(cnn))
        model.add(LSTM(10))
        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 3:
        bilstm = Sequential()
        n_timesteps = 1001
        bilstm.add(BatchNormalization())
        bilstm.add(Bidirectional(LSTM(64), input_shape=(n_timesteps, 10)))
        bilstm.add(Dense(5, activation='softmax'))
        return bilstm

    elif version == 4:
        # 60%+ val accuracy @ 20 epochs, adam, batch_size=32
        model = Sequential()
        model.add(BatchNormalization(momentum=0.98,input_shape=(1001, 10)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 5:
        # 74%+ val accuracy @ 20 epochs, adam, batch_size=32
        model = Sequential()
        model.add(BatchNormalization(momentum=0.98,input_shape=(1001, 10)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
        model.add(Attention(1001))
        model.add(Dropout(.2))
        model.add(Dense(400, activation='elu'))
        model.add(Dropout(.2))
        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 6:
        # 78% val accuracy @ 20 epochs, adam, batch_size=32
        model = Sequential()
        model.add(BatchNormalization(momentum=0.98,input_shape=(1001, 10)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
        model.add(Attention(1001))
        model.add(Dropout(.33))
        model.add(Dense(400, activation='elu'))
        model.add(Dropout(.33))
        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 7:
        # 80% val accuracy @ 20 epochs, adam, batch_size=32
        model = Sequential()
        model.add(BatchNormalization(momentum=0.98,input_shape=(1001, 10)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
        model.add(Attention(1001))
        model.add(Dropout(.33))
        model.add(Dense(400, activation='elu'))
        model.add(Dropout(.33))
        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 8:

        input_spectrum = Input(shape=(1001, 10, 1))

        branch_cnn = BatchNormalization()(input_spectrum)
        branch_cnn = Conv2D(32, kernel_size=3, activation='relu')(branch_cnn)
        branch_cnn = MaxPooling2D(pool_size=(2,1), strides=(2,1),
                                  padding='valid')(branch_cnn)
        branch_cnn = Flatten()(branch_cnn)
        branch_cnn = Dropout(.33)(branch_cnn)
        branch_cnn = Dense(400, activation='relu')(branch_cnn)

        branch_lstm = Reshape((1001, 10))(input_spectrum)
        branch_lstm = BatchNormalization(momentum=0.98)(branch_lstm)
        branch_lstm = Bidirectional(CuDNNGRU(128, return_sequences=True))(branch_lstm)
        branch_lstm = Bidirectional(CuDNNGRU(128, return_sequences=True))(branch_lstm)
        branch_lstm = Attention(1001)(branch_lstm)
        branch_lstm = Dropout(.33)(branch_lstm)
        branch_lstm = Dense(400, activation='elu')(branch_lstm)

        branch_out = concatenate([branch_cnn, branch_lstm])
        branch_out = Dropout(.33)(branch_out)

        model_out = Dense(5, activation="softmax")(branch_out)

        model = Model(inputs=input_spectrum, outputs=model_out)

        return model

    elif version == 9:
        # 80% val accuracy @ 30 epochs, adam, batch_size=32
        model = Sequential()
        model.add(Reshape((1001, 10)))
        model.add(BatchNormalization(momentum=0.98,input_shape=(1001, 10)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
        model.add(Attention(1001))
        model.add(Dropout(.35))
        model.add(Dense(500, activation='elu'))
        model.add(Dropout(.35))
        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 10:
        # 78% val accuracy @ 30 epochs, adam, batch_size=32

        input_spectrum = Input(shape=(1001, 10, 1))
        branch_in = Reshape((1001, 10))(input_spectrum)
        branch_in = BatchNormalization(momentum=0.98)(branch_in)

        branch_1 = Bidirectional(CuDNNGRU(128, return_sequences=True))(branch_in)
        branch_1 = Bidirectional(CuDNNGRU(128, return_sequences=True))(branch_1)
        branch_1 = Attention(1001)(branch_1)
        branch_1 = Dropout(.5)(branch_1)
        branch_1 = Dense(500, activation='elu')(branch_1)

        branch_2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(branch_in)
        branch_2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(branch_2)
        branch_2 = Attention(1001)(branch_2)
        branch_2 = Dropout(.5)(branch_2)
        branch_2 = Dense(500, activation='elu')(branch_2)

        branch_out = concatenate([branch_1, branch_2])
        branch_out = Dropout(.33)(branch_out)

        model_out = Dense(5, activation="softmax")(branch_out)

        model = Model(inputs=input_spectrum, outputs=model_out)

        return model

    elif version == 11:
        # 84% val accuracy @ 35 epochs, adam, batch_size=64, set_05
        model = Sequential()
        model.add(Reshape((1001, 10), input_shape=(1001, 10, 1)))
        model.add(BatchNormalization(momentum=0.98, input_shape=(1001, 10)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
        model.add(Attention(1001))
        model.add(Dropout(.5))
        model.add(Dense(500, activation='elu'))
        model.add(Dropout(.5))
        model.add(Dense(5, activation='softmax'))

        return model

    elif version == 12:
        # _% val accuracy @ 35 epochs, adam, batch_size=64, set_05
        model = Sequential()
        model.add(Reshape((1001, 10), input_shape=(1001, 10, 1)))
        model.add(BatchNormalization(momentum=0.98, input_shape=(1001, 10)))
        model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        model.add(Attention(1001))
        model.add(Dropout(.5))
        model.add(Dense(500, activation='elu'))
        model.add(Dropout(.5))
        model.add(Dense(5, activation='softmax'))

        return model


    else:
        return None


train_path = 'datagen/data/train_01.pkl'
test_path = 'datagen/data/test_01.pkl'

learning_rate = .0005
momentum = .9
decay = 1e-6
loss = "categorical_crossentropy" #"poisson",
batch_size = 64
n_epochs = 35
optimizer = 'adam'
#optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay)
use_version = 12
save_name = "model-version" + str(use_version)
dataset_name = "set_05"


if __name__ == "__main__":
    main()
