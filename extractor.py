import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras

def first_x_sec(seconds, filepath, timecol):
    ''' param: seconds, int, filter out the first seconds of the game
        param: filepath, string, the filepath of the csv datafile
        param: timecol, int, the index of the column that contains time
    '''
    f = open(filepath)
    lines = f.readlines()
    lines = [ line.split(',') for line in lines ]
    if seconds is None:
        return np.array(lines[1:])
    else:
        return np.array(list(filter(lambda line: int(line[timecol])<=seconds, lines[1:])))

# First 5 min: 5*60=300, col 2 is time first line is header
#ability_upgrade = first_x_sec(300, 'dataset/ability_upgrades.csv', 2)
#objectives = first_x_sec(300, 'dataset/objectives.csv', 7)
player_time = first_x_sec(300, 'dataset/player_time.csv', 1)

## Match file contains more than just the label
# column 10 contains the True/False value of Radiant win
matches = first_x_sec(None, 'dataset/match.csv', None)

labels = np.array([int(match[9] == 'True') for match in matches])
print('All the labels')
print(labels)

print('Shape of player time matrix')
print('Contains all attributes of matches, the first minutes')
print(player_time.shape)
#print(player_time[1:10])
player_time = player_time.astype(np.int)

## Separate data matches. Each index in data is one match. One match is a 32*x.
## x is the number of logs. We have data every 60 secods, so for 5 min x=5
data = [np.empty(shape=(32,0), dtype=np.int)]*50000
for x in player_time:
    data[x[0]] = np.concatenate((data[x[0]], x.reshape(32,1)), axis=1)
#Print the first match log
print('The pre prossessed data, just the second match')
print(data[1])

#Filter out the end state from data. 
# This will be a matrix with the game state afer x seconds in each column
# the first row is time
# following by gold, last hits and xp for each player
end_5_min_games = np.array([game[2:,-1] for game in data]).T
#print the data to look at it
print('Example data table. First 10 matches. Game state at the end of 5 min')
print(end_5_min_games[:,:10])
# the shape should be 31, 50000. 
print('The shape of the match data table, first row is the time column, which is always 300')
print(end_5_min_games.shape)


model = Sequential()

model.add(Dense(units=100, input_dim=30))
model.add(Activation('relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.0, nesterov=False),
             metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])

print('Shapes of labels')
print(labels.shape)
from keras.utils.np_utils import to_categorical
labels_binary = to_categorical(labels)
print('example of how the labels look like')
print(labels[:10])
print('Re-structure the labels so that it is two dimensions, with prob value of each team winning')
print(labels_binary.shape)
print('example of how the binary labels look like')
print(labels_binary[:10])

# separate into train and test
x_train = end_5_min_games[:,:40000]
y_train = labels_binary[:40000,:]
x_test = end_5_min_games[:,40000:]
y_test = labels_binary[40000:,:]
print(y_test.shape)

res = model.fit(x_train.T, y_train, epochs=50, batch_size=1000, validation_split=0.8)
# Test loss and accuracy
loss_and_metrics = model.evaluate(x_test.T, y_test)
print(loss_and_metrics)

