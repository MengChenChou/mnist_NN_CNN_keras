# coding: utf-8
import keras

# 匯入MNIST資料
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import pandas as pd

# 載入模型
model = load_model('C:/Users/User/PycharmProjects/deep_ten_kera/tmp_param_keras/model_KerasMnistCNN.h5')

# 顯示出模型摘要
model.summary()

# 讀取MNIST資料為Tuple形式, x_train為影像資料, y_train為標籤資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 保持原始數據, 之後混淆矩陣 做比較用
label_y_test = y_test

# 圖片正規劃
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

# 輸入的x變成60,000*784的數據，然後除以255進行標準化，每個像素都是在0到255之間的(顏色階數從0~255)，標準化之後就變成了0到1之間。
x_test /= 255

# 把y變成了one-hot的形式
y_test = keras.utils.to_categorical(y_test,  num_classes=10)

# 驗證模型
score = model.evaluate(x_test, y_test, verbose=0)
# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict_classes(x_test)
print(x_test.shape)
print(predictions)
"""
# 顯示出前15個測試影像, 預測結果, 與原始答案
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.title("pred.={} label={}".format(predictions[i],np.argmax(y_test[i])))
    plt.imshow(x_test[i].reshape(28, 28), cmap=None)
plt.show()
"""
print("Error prediction:")
errorList = []
for i in range(len(predictions)):
    if predictions[i] != np.argmax(y_test[i]):
        print("Image[%d] : label=%d, but prediction=%d" % (i, np.argmax(y_test[i]), predictions[i]))
        errorList.append(i)
print("-----------------------")
print("total number of error prediction is %d" % len(errorList))


print("%s\n" % pd.crosstab(label_y_test, predictions, rownames=['label'], colnames=['predict']))