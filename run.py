import lstm
import time
import numpy as np
import matplotlib.pyplot as plt

def plot_results(predicted_data, true_data):


    plt.plot(true_data, 'o-',label='True Data')
    plt.plot(predicted_data,'*-', label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':

    global_start_time = time.time()
    epochs  = 1
    seq_len = 50

    print('> Loading data... ')

    X_train, y_train, X_test, y_test = lstm.load_data('spc.csv', seq_len, True)
    #X_train, y_train, X_test, y_test = lstm.load_data('sinwave.csv', seq_len, False)

    print('> Data Loaded. Compiling...')

    model = lstm.build_model([1, 50, 100, 1])

    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

    #predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    #predicted = lstm.predict_point_by_point(model, X_test)

    print(np.array(predicted).shape)
    print(y_test.shape)
    print('Training duration (s) : ', time.time() - global_start_time)
    #plot_results_multiple(predictions, y_test, 50)
    plot_results(predicted,y_test)