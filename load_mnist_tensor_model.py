import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
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

# 初始化所有參數
init = tf.global_variables_initializer()
# 載入模型加以應用
with tf.Session() as sess:
    sess.run(init)
    # 載入保存的模型
    saver.restore(sess, model_path)
    # 用test集測試模型的準確率，順便看模型是否存在
    print("test accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, Y: mnist.test.labels}))





