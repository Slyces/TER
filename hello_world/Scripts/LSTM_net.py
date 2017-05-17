import pandas
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import warnings
import sqlite3 as sql
import time
from django_world.settings import DATABASE_PATH
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_data():
    with sql.connect(DATABASE_PATH) as con:
        data = pandas.read_sql("SELECT val FROM historic_snp;", con)

    data = np.array(data)
    return data


def sliding_window(data, seq_len, normalise):
    # print("data :\n", data)
    sequence_length = seq_len + 1
    result = []
    # print(data)
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
        if normalise:
            result[-1] = normalise_window(result[-1])

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]

    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]


def normalise_window(window):
    p_0 = window[0]
    normalised_window = [((float(p) / float(p_0)) - 1) for p in window]
    return normalised_window


def revert_window(history, window, seq_len):
    p_0 = history[-seq_len]
    return [(float(p_0) * (float(p) + 1)) for p in window]


def build_model(layers, drop=0.2):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(drop))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(drop))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(len(data) // prediction_len):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data)
        plt.legend()
    plt.show()


def plot_results_unique(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    plt.plot(predicted_data, label='predictions')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Datas sampling
    data = load_data()
    seq = 50
    X_train, y_train, X_test, y_test = sliding_window(load_data(), seq, True)

    # Model building
    model = build_model([1, seq, 200, 1], drop=0.2)

    # Training
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=1,
        validation_split=0.05)

    # Multiplot
    predictions = predict_sequences_multiple(model, X_test, seq, seq)
    plot_results_multiple(predictions, y_test, seq)

    # Single plot
    '''
    predictions = predict_point_by_point(model, X_test)

    reverted_hist = revert_window(data, y_test, seq)
    reverted_prediction = revert_window(data, predictions, seq)
    plot_results_unique(reverted_prediction,
                        reverted_hist)

    with sql.connect(DATABASE_PATH) as con:
        curs = con.cursor()
        for i in range(len(reverted_hist)):
            curs.execute("""INSERT INTO lstm_single (historic, predicted)
                            VALUES (?, ?);""", (reverted_hist[i], reverted_prediction[i]))
        con.commit()
    '''
