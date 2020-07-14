import keras
# 匯入MNIST資料
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import pandas as pd
# 載入模型
model = load_model('C:/Users/User/PycharmProjects/ai_deep/tmp_param_keras/model_KerasMnistMLP.h5')
# 顯示出模型摘要
model.summary()

# 讀取MNIST資料為Tuple形式, x_train為影像資料, y_train為標籤資料
# X shape (60,000 28x28), y shape (10,000, )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 保持原始數據, 之後混淆矩陣 做比較用
keep_y_test = y_test

# 將輸入的資料正規劃
x_test = x_test.reshape(10000, 784).astype('float32')

# 輸入的x變成60,000*784的數據，然後除以255進行標準化，每個像素都是在0到255之間的(顏色階數從0~255)，標準化之後就變成了0到1之間。
x_test /= 255

#把lebel變成one-hot的形式,# 數字為0~9所以共10個類別
y_test = keras.utils.to_categorical(y_test, num_classes=10)


# 驗證模型
score = model.evaluate(x_test, y_test, verbose=0)
# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 印出所有預測結果
predictions = model.predict_classes(x_test)
print(predictions.shape)  #(10000,)
print("All predictions: ")
print(predictions)
"""
# 顯示出前15個測試影像, 預測結果, 與原始答案
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.title("pred.={} label={}".format(predictions[i],np.argmax(y_test[i])))
    plt.imshow(x_test[i].reshape(28, 28))
plt.show()
"""
# 印出辨識錯誤的  混淆矩陣
errorList = []
for i in range(len(predictions)):
    if predictions[i] != np.argmax(y_test[i]):
        print("Image[%d] : label=%d, but prediction=%d" % (i, np.argmax(y_test[i], axis=0), predictions[i]))
        errorList.append(i)
print("-----------------------")
print("total number of error prediction is %d" % len(errorList))
print("-----------------------")
# 印出比較表
print("%s\n" % pd.crosstab(keep_y_test, predictions, rownames=['label'], colnames=['predict']))