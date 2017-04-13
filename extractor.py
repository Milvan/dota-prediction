import numpy as np

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
player_time = first_x_sec(500, 'dataset/player_time.csv', 1)


## Match file contains more than just the label
labels = first_x_sec(None, 'dataset/match.csv', None)

print(player_time.shape)
#print(player_time[1:10])
player_time = player_time.astype(np.int)

## Separate data matches. Each index in data is one match. One match is a 32*x.
## x is the number of logs. We have data every 60 secods, so for 5 min x=5
data = [np.empty(shape=(32,0), dtype=np.int)]*50000
for x in player_time:
    data[x[0]] = np.concatenate((data[x[0]], x.reshape(32,1)), axis=1)

#Print the first match log 
print(data[0])
