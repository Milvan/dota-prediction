import numpy as np

games = np.load('/tmp/game_data.npy')
print (games.shape)

labels = np.load('/tmp/game_labels.npy')
print (labels.shape)


#labels_binary_length_corrected = labels.reshape((300000, 2))
labels_binary_length_corrected = labels

number_of_features_to_use = games.shape[2]

data = np.zeros((games.shape[0],games.shape[1], number_of_features_to_use))
for i in range(games.shape[0]):
    for j in range(games.shape[1]):
        data[i, j] = games[i,j][:number_of_features_to_use]
print (data.shape)


val_split = int(len(data) * 0.9)
train_split = int(val_split * 0.8)
print(data.shape)
x_train = data[:train_split,:]
y_train = labels_binary_length_corrected[:train_split,:]
x_val = data[train_split:val_split,:]
y_val = labels_binary_length_corrected[train_split:val_split,:]
x_test = data[val_split:,:]
y_test = labels_binary_length_corrected[val_split:,:]

#x_train = np.reshape(x_train, x_train.shape + (1,))
#x_test = np.reshape(x_test, x_test.shape + (1,))

print(x_train.shape)

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.regularizers import l2
import keras
from sklearn.metrics import roc_auc_score


# batch eta lambda units

results = open('/tmp/dota_coarse_search.csv', 'a')
results.write("val_acc, auc, val_acc_certain, auc_certain, batch_size, lstm_size, dense_size, regularization lstm, eta\n")

batch_sizes = [2**y for y in range(5,10)]

i=0
while(True):
    i+=1

    print("Network number %i in training" % i)

    batch_size = np.random.choice(batch_sizes)
    #print("batch size: ", batch_size)
    lstm_size = np.random.randint(low=50, high=1000)
    #print("lstm size: ", lstm_size)
    regu = np.random.uniform(low=0, high=0.1)
    #print("regularization: ", regu)
    eta = np.random.uniform(low=1e-6, high=0.0035)
    #print("eta: ", eta)

    model = Sequential()

    model.add(LSTM(
        input_shape=(None, x_train.shape[2]),
        units=lstm_size,
        return_sequences=False,
        kernel_regularizer= l2(regu))
        )


    dense_size = np.random.randint(low=50,high=1500)
    #print("dense size: ", dense_size)
    model.add(Dense(
        units=dense_size))
    model.add(Activation('relu'))

    model.add(Dense(
        units=2))
    model.add(Activation('softmax'))
    #model.compile(loss='mse', optimizer='rmsprop')
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=eta), metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])

    model_checkpoint = keras.callbacks.ModelCheckpoint('/tmp/best_weights', monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.0, validation_data=(x_val, y_val), verbose=0, callbacks = [model_checkpoint])

    model.load_weights('/tmp/best_weights')


    # Test loss and accuracy
    loss_and_metrics1 = model.evaluate(x_val, y_val, verbose=0)


    ## Calculate AUC
    y_pred = model.predict_proba(x_val, verbose=0)
    score1 = roc_auc_score(y_val, y_pred)

    ## Remove uncertain predictions
    ind1 = y_pred[:,0] > 0.6
    ind2 = y_pred[:,1] > 0.6
    ind = ind1 + ind2
    y_pred2 = y_pred[ind,:]
    y_val2 = y_val[ind,:]
    x_val2 = x_val[ind,:]
    #print(ind)
    #print(y_pred2.shape)
    #print(y_pred.shape)
    score2 = 0 if len(y_pred2)==0 else roc_auc_score(y_val2, y_pred2)
    #print(score2)

    loss_and_metrics2 = [0,0,0] if len(y_pred2)==0 else model.evaluate(x_val2, y_val2, verbose=2)
    #print(loss_and_metrics)

    log = ",".join( list(map(str, [loss_and_metrics1[2], score1, loss_and_metrics2[2], score2, batch_size, lstm_size, dense_size, regu, eta, ]))) + "\n"
    print(log)
    results.write(log)
    results.flush()
    #print(loss_and_metrics)

results.close()

