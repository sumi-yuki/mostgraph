# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 23:28:30 2023
"""


#confirmed to work on tensorflow2.0

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] ='sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio']

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential

import tensorflowjs as tfjs

#正常者データー読み込み
#検証用データー として mostgraph_normal_test_data_numpy
#学習用データー として mostgraph_normal_train_data_numpy
#に入力
#検証用データー数は number_of_test = 20 で設定

# csvファイルの読み込み
mostgraphdata = pd.read_csv("female,male_controldata_smart.csv", encoding="utf-8")
# diagnosis(診断)の項目を削除
#mostgraphdata = mostgraphdata.drop("diagnosis", axis=1)
# 読み込みの確認
print("正常入力データー(pandas)")
print(mostgraphdata)

# pandas -> numpyへ変換
mostgraph_total_data_numpy = mostgraphdata.values
# 変換の確認
print("正常入力データー(numpy)")
print(mostgraph_total_data_numpy)
print(mostgraph_total_data_numpy.shape)
print(mostgraph_total_data_numpy[0])

#検証用データー数
number_of_test = 10

# データー数を取得
number_of_normal_mostgraphdata = mostgraph_total_data_numpy.shape[0]
# 確認
print("正常データー数")
print(number_of_normal_mostgraphdata)

# ランダムに並べ替える
np.random.shuffle(mostgraph_total_data_numpy)
# 変換の確認
print("正常シャッフル後入力データー(numpy)")
print(mostgraph_total_data_numpy)
print(mostgraph_total_data_numpy.shape)
print(mostgraph_total_data_numpy[0])

#検証用データーの抽出
mostgraph_normal_test_data_numpy = mostgraph_total_data_numpy[0:number_of_test]
# 確認
print("正常検証用データー")
print(mostgraph_normal_test_data_numpy)
print(mostgraph_normal_test_data_numpy.shape)
print(mostgraph_normal_test_data_numpy[0])

#学習用データーの抽出
mostgraph_normal_train_data_numpy = mostgraph_total_data_numpy[number_of_test:number_of_normal_mostgraphdata]
# 確認
print("正常学習用データー")
print(mostgraph_normal_train_data_numpy)
print(mostgraph_normal_train_data_numpy.shape)
print(mostgraph_normal_train_data_numpy[0])

#学習用データのかさまし、x軸方向に1倍、y軸方向に5倍する
mostgraph_normal_train_data_numpy_extended=np.tile(mostgraph_normal_train_data_numpy,(5, 1))



#疾患データー読み込み
#検証用データー として mostgraph_abnormal_test_data_numpy
#学習用データー として mostgraph_abnormal_train_data_numpy
#に入力
#検証用データー数は number_of_test = 20 で設定

# csvファイルの読み込み
mostgraphdata = pd.read_csv("female,male_patientsdata_smart.csv", encoding="utf-8")
# diagnosis(診断)の項目を削除
#mostgraphdata = mostgraphdata.drop("diagnosis", axis=1)
# 読み込みの確認
print("疾患入力データー(pandas)")
print(mostgraphdata)

# pandas -> numpyへ変換
mostgraph_total_data_numpy = mostgraphdata.values
# 変換の確認
print("疾患入力データー(numpy)")
print(mostgraph_total_data_numpy)
print(mostgraph_total_data_numpy.shape)
print(mostgraph_total_data_numpy[0])

# ランダムに並べ替える
np.random.shuffle(mostgraph_total_data_numpy)
# 変換の確認
print("疾患シャッフル後入力データー(numpy)")
print(mostgraph_total_data_numpy)
print(mostgraph_total_data_numpy.shape)
print(mostgraph_total_data_numpy[0])

# データー数を取得
number_of_abnormal_mostgraphdata = mostgraph_total_data_numpy.shape[0]
# 確認
print("疾患データー数")
print(number_of_abnormal_mostgraphdata)

#検証用データーの抽出
mostgraph_abnormal_test_data_numpy = mostgraph_total_data_numpy[0:number_of_test]
# 確認
print("疾患検証用データー")
print(mostgraph_abnormal_test_data_numpy)
print(mostgraph_abnormal_test_data_numpy.shape)
print(mostgraph_abnormal_test_data_numpy[0])

#学習用データーの抽出
mostgraph_abnormal_train_data_numpy = mostgraph_total_data_numpy[number_of_test:number_of_abnormal_mostgraphdata]
# 確認
print("疾患学習用データー")
print(mostgraph_abnormal_train_data_numpy)
print(mostgraph_abnormal_train_data_numpy.shape)
print(mostgraph_abnormal_train_data_numpy[0])

#5) 神経回路の作成
model = Sequential()

model.add(Dense(units=128, input_shape=(11,)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(units=64))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(units=64))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(units=32))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(units=16))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=["accuracy"])
model.summary() 




#学習用データー統合
mostgraph_train_data_numpy = np.concatenate([mostgraph_normal_train_data_numpy_extended, mostgraph_abnormal_train_data_numpy])
# 確認
print("結合後学習用データー")
print(mostgraph_train_data_numpy)
print(mostgraph_train_data_numpy.shape)
print(mostgraph_train_data_numpy[0])
# ランダムに並べ替える
np.random.shuffle(mostgraph_train_data_numpy)
# 変換の確認
print("結合後ランダム学習用データー")
print(mostgraph_train_data_numpy)
print(mostgraph_train_data_numpy.shape)


#Correct for data imbalance
number_of_normal_mostgraphdata_extended = mostgraph_normal_train_data_numpy_extended.shape[0]
number_of_normal_mostgraphdata = number_of_normal_mostgraphdata_extended - number_of_test
#number_of_normal_mostgraphdata = number_of_normal_mostgraphdata - number_of_test
number_of_abnormal_mostgraphdata = number_of_abnormal_mostgraphdata - number_of_test
initial_bias = np.log([number_of_abnormal_mostgraphdata / number_of_normal_mostgraphdata])
print("Initial bias: {:.5f}".format(initial_bias[0]))

number_of_total_train_mostgraphdata = number_of_normal_mostgraphdata + number_of_abnormal_mostgraphdata
weight_for_0 = (1 / number_of_normal_mostgraphdata) * (number_of_total_train_mostgraphdata) / 2.0
weight_for_1 = (1 / number_of_abnormal_mostgraphdata) * (number_of_total_train_mostgraphdata) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))


labels_train = mostgraph_train_data_numpy[ : , 0:1]
data_train = mostgraph_train_data_numpy[ : , 1:12]

#4) データの確認
print(labels_train.shape)
#print(labels_train)

print(data_train.shape)
#print(data_train)


#6) 学習実行　epoch数を繰り返す　0.5割は検証用に用い学習には使わない
model_history = model.fit(data_train, labels_train, class_weight=class_weight, batch_size = 32, epochs = 2500, validation_split = 0.01)


#7) 学習過程をグラフ表示する
history_dict = model_history.history
# Loss(正解との誤差)をloss_valuesに入れる
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
# 正確度をaccに入れる
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
# 1からepoch数までのリストを作る
epochlist = range(1, len(loss_values) +1)
#  正確度のグラフを作る
# 'b'は青い線
plt.plot(epochlist, acc, 'bo', label='Accuracy at training')
plt.plot(epochlist, val_acc, 'b', label='Accuracy at validation')
#  Loss(正解との誤差)のグラフを作る
# 'ro'は赤い点  https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
plt.plot(epochlist, loss_values, 'ro', label='Loss at training')
plt.plot(epochlist, val_loss_values, 'r', label='Loss at validation')
#  タイトル
plt.title('学習回数と正確度、誤差')
plt.ylabel('青点は学習時正解率、青線は検証時正解率、赤点は学習時誤差、赤線は検証時誤差')
plt.xlabel('学習回数(epoch数)')
plt.legend()
#  グラフを表示する
plt.show()



#print("正常検証用データー")
#print(mostgraph_normal_test_data_numpy)
#print("疾患検証用データー")
#print(mostgraph_abnormal_test_data_numpy)

labels_test = mostgraph_normal_test_data_numpy[ : , 0:1]
data_test = mostgraph_normal_test_data_numpy[ : , 1:14]
loss, accuracy = model.evaluate(data_test, labels_test)
print("Accuracy for normal= {:.2f}".format(accuracy))
predicted = model.predict(data_test)
print("predicted for normal:", predicted)
#print(data_test)

labels_test = mostgraph_abnormal_test_data_numpy[ : , 0:1]
data_test = mostgraph_abnormal_test_data_numpy[ : , 1:14]
loss, accuracy = model.evaluate(data_test, labels_test)
print("Accuracy for abnormal= {:.2f}".format(accuracy))
predicted = model.predict(data_test)
print("predicted for abnormal:", predicted)
#print(data_test)

#モデルの保存
#python用
model.save("deeplearing_model")
#javascript用
#tfjs_target_dir = "./"
#tfjs.converters.save_keras_model(model, tfjs_target_dir)
#import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "./mostgraph_model")
