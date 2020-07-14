import keras

#匯入MNIST資料
from keras.datasets import mnist
#使用Sequential模型
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

# 每一批次讀入128張資料
batch_size = 128

# 數字為0~9所以共10個類別
num_classes = 10

# 使用反向傳播法進行訓練，總共訓練20次
epochs = 10

# 讀取MNIST資料為Tuple形式, x_train為影像資料, y_train為標籤資料
# X shape (60,000 28x28), y shape (10,000, )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  #(60000, 28, 28)
print(y_train.shape)  #(60000,)
print(x_test.shape)   #(10000, 28, 28)
print(y_test.shape)   #(10000,)
# 將輸入的資料正規劃
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

# 輸入的x變成60,000*784的數據，然後除以255進行標準化，每個像素都是在0到255之間的(顏色階數從0~255)，標準化之後就變成了0到1之間。
x_train /= 255
x_test /= 255

#把lebel變成one-hot的形式
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 一層一層的去建立神經層
model = Sequential()

# 加入全連接層, 輸入為任意筆數的784維資料(28*28), 256個神經元
model.add(Dense(input_dim=784,
                units=256,  # <--可以自己自由定義神經元的數量
                kernel_initializer='normal',
                bias_initializer='zeros',
                activation='relu', #relu激勵函數
                name='hidden1'
                ))

# 加入dropout隨機在訓練時關閉輸入單元與權重的影響, 防止過擬合 , 0.2表示有2成輸入單元被關閉
model.add(Dropout(0.2))

# 加入第二層全連接層, 256個神經元
model.add(Dense(units=256,
                kernel_initializer='normal',
                bias_initializer='zeros',
                activation='relu',
                name='hidden2'
                ))

model.add(Dropout(0.2))

# 加入第三層輸出層為全連接層, 10個神經元
model.add(Dense(units=10,
                kernel_initializer='normal',
                bias_initializer='zeros',
                activation='softmax',
                name='output'
                ))
# 顯示出模型摘要
model.summary()

# 損失函數用交叉熵
# 優化器用RMSprop
# metrics，放入需要觀察的accuracy
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#將epoch的訓練結果保存在csv文件中
logger = CSVLogger('C:/Users/User/PycharmProjects/ai_deep/tmp_param_keras/model_KerasMnistMLP.log')

#當監測值val_loss不再改善時，如發現損失沒有下降，則經過3次epoch後停止訓練。
estop = EarlyStopping(monitor='val_loss', patience=3)


#模型訓練
# verbose：日誌顯示
# verbose = 0為不在標準輸出流輸出日誌信息
# verbose = 1為輸出進度條記錄
# verbose = 2為每個epoch輸出一行記錄
# 注意：默認為1

#validation_split：0~1之間的浮點數，用來指定訓練集當作驗證集的比例。
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1, # verbose = 1為輸出進度條記錄
                 validation_split=0.1,
                 callbacks=[logger, estop])
# 進行模型評估
# verbose：日誌顯示
# verbose = 0為不在標準輸出流輸出日誌信息
# verbose = 1為輸出進度條記錄
score = model.evaluate(x_test, y_test, verbose=1)
print('test loss:', score[0])
print('test acc:', score[1])

# 顯示acc學習結果
accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss學習結果
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
model.save('C:/Users/User/PycharmProjects/ai_deep/tmp_param_keras/model_KerasMnistMLP.h5')

del model

