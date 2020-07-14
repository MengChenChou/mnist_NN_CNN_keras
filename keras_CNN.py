# coding: utf-8
import keras

# 匯入MNIST資料
from keras.datasets import mnist
# 使用Sequential模型
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

# 每一批次讀入128張資料
batch_size = 200

# 數字為0~9所以共10個類別
num_classes = 10

# 使用反向傳播法進行訓練，總共訓練20次
epochs = 10

# 讀取MNIST資料為Tuple形式, x_train為影像資料, y_train為標籤資料
# X shape (60,000 28x28), y shape (10,000, )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 將輸入的資料正規劃   #28x28x1一通道的灰階圖
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

# 輸入的x變成60,000*784的數據，然後除以255進行標準化，每個像素都是在0到255之間的(顏色階數從0~255)，標準化之後就變成了0到1之間。
x_train /= 255
x_test /= 255

# 把y變成了one-hot的形式
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 印出形狀
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# 起手式
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'
                ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

# 顯示出模型摘要
model.summary()

# 損失函數用交叉熵,
# 優化器用RMSprop,
# metrics，裡面可以放入accuracy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#將epoch的訓練結果保存在csv文件中
logger = CSVLogger('C:/Users/User/PycharmProjects/deep_ten_kera/tmp_param_keras/log_KerasMnistCNN.log')

#當監測值val_loss不再改善時，如發現損失沒有下降，則經過3個epoch後停止訓練。
estop = EarlyStopping(monitor='val_loss', patience=3)

#模型訓練
#validation_split：0~1之間的浮點數，用來指定訓練集當作驗證集的比例。
# verbose：日誌顯示
# verbose = 0為不在標準輸出流輸出日誌信息
# verbose = 1為輸出進度條記錄
# verbose = 2為每個epoch輸出一行記錄
# 注意：默認為1
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_split=0.2,
                 callbacks=[logger,estop])

# 進行模型評估
score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])

# 顯示acc訓練結果
accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss訓練結果
loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(range(len(loss)), loss, marker='.', label='loss(training data)')
plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 存儲模型與權重
model.save('C:/Users/User/PycharmProjects/deep_ten_kera/tmp_param_keras/model_KerasMnistCNN.h5')

del model

