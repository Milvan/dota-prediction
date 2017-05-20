import numpy as np
data = np.load('/tmp/game_data.npy')
labels = np.load('/tmp/game_labels.npy')

data = data[:,-1,:]


val_split = int(len(data) * 0.8)
train_split = int(val_split * 0.7)
print(data.shape)
x_train = data[:train_split,:]
y_train = labels[:train_split,:]
x_val = data[val_split:,:]
y_val = labels[val_split:,:]
x_test = data[train_split:val_split,:]
y_test = labels[train_split:val_split,:]

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
import keras
from sklearn.metrics import roc_auc_score



results = open('/tmp/dota_coarse_search.csv', 'a')
results.write("val_acc, auc, val_acc_certain, auc_certain, batch_size, lstm_size, dense_size, regularization lstm, eta\n")

batch_sizes = [2**y for y in range(5,10)]

i=0
while(True):
    i+=1

    print("Network number %i in training" % i)

    batch_size =100# np.random.choice(batch_sizes)
    #print("batch size: ", batch_size)
    lstm_size = 0;
    dense_size =500# np.random.randint(low=50, high=2000)
    #print("lstm size: ", lstm_size)
    regu = 0.01# np.random.uniform(low=0, high=0.1)
    #print("regularization: ", regu)
    eta = 0.001# np.random.uniform(low=1e-6, high=0.035)
    #print("eta: ", eta)

    model = Sequential()

    model.add(Dense(units=dense_size, input_dim=data.shape[1], kernel_regularizer=l2(regu)))
    model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.RMSprop(lr=eta),
                                      metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])

    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.0, validation_data=(x_val, y_val), verbose=0, callbacks = [])



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

    score2 = 0
    loss_and_metrics2 = [0,0,0]

    certain_count = len(y_pred2)
    if len(np.unique(y_val2[:,0])) >1:
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

