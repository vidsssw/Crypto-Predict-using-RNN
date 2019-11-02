import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
import os
import time
import numpy as np
from sklearn import preprocessing
from collections import deque
import pandas as pd
import random

to_predict="LTC-USD"
def classify(a,b):
    if float(b)>float(a):
        return 1
    else:
        return 0
def balance(df):
    buy=[]
    sell=[]

    for seq,target in df:
        if target==0:
            sell.append([seq,target])
        else:
            buy.append([seq,target])
    random.shuffle(buy)
    random.shuffle(sell)

    m=min(len(buy),len(sell))

    buy=buy[:m]
    sell=sell[:m]

    df=buy+sell

    return df

def preproc(df):
    df=df.drop("future",1)
    for col in df.columns:
        if col!="target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    seq_data=[]
    prev_days=deque(maxlen=60)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if(len(prev_days)==60):
            seq_data.append([np.array(prev_days),i[-1]])
    random.shuffle(seq_data)

    seq_data=balance(seq_data)
    random.shuffle(seq_data)
    X=[]
    Y=[]
    for seq,target in seq_data:
        X.append(seq)
        Y.append(target)

    return np.array(X),Y














names=['LTC-USD','ETH-USD','BTC-USD','BCH-USD']
main_df=pd.DataFrame()
for name in names:
    name = name.split('.csv')[0]
    df = pd.read_csv(f'{name}.csv', names=['time', 'low', 'high', 'open', 'close','volume'])
    df.rename(columns={"close": f'{name}_close', "volume": f'{name}_volume'},inplace=True)
    df.set_index("time",inplace=True)

    df=df[[f'{name}_close',f'{name}_volume']]

    if(len(main_df)==0):
        main_df=df

    else:
        main_df=main_df.join(df)

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)

main_df['future']=main_df[f'{to_predict}_close'].shift(-3)
main_df['target']=list(map(classify,main_df[f'{to_predict}_close'],main_df['future']))

times = sorted(main_df.index.values)
last = sorted(main_df.index.values)[-int(0.05*len(times))]

test_df=main_df[(main_df.index>=last)]
main_df=main_df[(main_df.index<last)]

train_x,train_y=preproc(main_df)
test_x,test_y=preproc(test_df)
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


# Train model
history = model.fit(
    train_x, train_y,
    batch_size=64,
    epochs=1,
    validation_data=(test_x, test_y),

)

# Score model
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



