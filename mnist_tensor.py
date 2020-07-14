import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
# 載入mnist資料
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
# 因為一張照片原始為(784, )所以我們要寫一個韓式轉換成像數，才能重現圖片
def data_to_matrix(data):
    return data.reshape(28, 28)
# 測試一張圖片
matrix = data_to_matrix(mnist.train.images[1])

plt.figure()
plt.imshow(matrix)
plt.show()
"""

# 開始訓練模型
x = tf.placeholder(tf.float32, shape=[None, 784]) # 輸入一張照片
Y = tf.placeholder(tf.float32, shape=[None, 10])

# 參數
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_pre = tf.matmul(x, W)+b

# 設定超參數
lr = 0.5
batch_size = 1000
epochs = 1000
epoch_list = []
accuracy_list = []
loss_list = []

# 設定loss及梯度更新方法
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pre))
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# train的正確率
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(Y, 1)) # 1列找最大
cp = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(cp)

# 模型參數存放
model_path = "C:/Users/User/PycharmProjects/ai_deep/tmp_param/mnist_tensor.ckpt"
saver = tf.train.Saver()

# 以上皆為模型設定，下面將寫序直接做模型訓練

#初始化所有參數
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size) # 每次隨機挑選batch_size的樣本數量
        _, loss_, cp_, accuracy_ = sess.run([train, loss, cp, accuracy], feed_dict={x: batch_x, Y: batch_y})
        epoch_list.append(epoch)
        accuracy_list.append(accuracy_)
        loss_list.append(loss_)

        if epoch % 500 == 0:
            print("accuracy={} loss={} epochs={}".format(accuracy_, loss_, epoch))

            plt.subplot(1, 2, 1)
            plt.plot(epoch_list, accuracy_list, lw=2)
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.title("train set:lr={} batch_size={} epochs={}".format(lr, batch_size, epochs))

            plt.subplot(1, 2, 2)
            plt.plot(epoch_list, loss_list, lw=2)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("train set:lr={} batch_size={} epochs={}".format(lr, batch_size, epochs))
            plt.show()
    print("訓練結束")
    # 將模型參數保存
    save_path = saver.save(sess, model_path)
    print("模型保存在:{}".format(save_path))
    sess.run(accuracy, feed_dict={x: mnist.test.images, Y: mnist.test.labels})








