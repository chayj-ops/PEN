import numpy as np
import tensorflow as tf
import random
import scipy.misc


##HPU test용
batch_size = 100  # batch size
num_category = 10  # total categorical factor
num_cont = 2  # total continuous factor
num_dim = 33  # total latent dimension
keep_prob = 0.7
#
# create generator
#

data_main = np.load('data_main.npy')
data_noise = np.load('data_noise.npy')
data_zero = np.load('data_zero.npy')

data_main = data_main[:, :, :, np.newaxis]
data_zero = data_zero[:, :, :, np.newaxis]
data_noise_mod = np.zeros((len(data_noise), 784))
for s in range(len(data_noise)):
    for ss in range(784):
        if ss == 0 or ss % 2 == 0:
            data_noise_mod[s, ss] = data_noise[s, 10]
        else:
            data_noise_mod[s, ss] = data_noise[s, 11]

# noise_box = np.zeros((batch_size, 784*10))
# for s in range(batch_size):
#  randomIndex = random.randrange(0, batch_size)
#  noise_box[s]=data_noise_mod[randomIndex]

print(data_main.shape)
print(data_noise.shape)
print(data_zero.shape)

ori = tf.placeholder(tf.float32, [None, 28, 28, 1], name='ori')
y = tf.placeholder(tf.float32, [None, 28, 28, 1], name='y')
z = tf.placeholder(tf.float32, [None, 784], name='z')


###########################
# 1이미지를 입력받아야 한다.
con_path = "1.png"

context_img = scipy.misc.imread(con_path).astype(np.float)
context_img = context_img/255



context_img1 = (context_img)[:,:,np.newaxis]
context_img2 = context_img1[np.newaxis,:,:,:]
print(context_img)
##########################




# CNN 모델을 정의합니다.
#def build_CNN_classifier(ori, z):
# 입력 이미지
x_image = ori
z = z

W_conv1 = tf.Variable(tf.truncated_normal(([4, 4, 1, 64]), stddev=0.1, name='C_conv1'), name='W_conv1')
b_conv1 = tf.Variable(tf.constant(0.1, shape=([64])), name='b_conv1')
L1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='Con1') + b_conv1, name='Relu1')

L2 = tf.contrib.layers.flatten(L1)

#L3 = tf.layers.dense(L2, 10000, name='L3')
W_den3 = tf.Variable(tf.truncated_normal(([28*28*64, 1000]), stddev=0.1, name='C_den3'), name='W_den3')
b_den3 = tf.Variable(tf.constant(0.1, shape=([1000]), name='const3'), name='b_den3')
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W_den3, name='Den3'), b_den3, name='matmul3'), name='Relu3')

L4 = tf.concat([L3, z], axis=1, name='L4')

#L5 = tf.layers.dense(L4, 7 * 7 * 64, name='L5')
W_den5 = tf.Variable(tf.truncated_normal(([1784, 7*7*64]), stddev=0.1, name='C_den5'), name='W_den5')
b_den5 = tf.Variable(tf.constant(0.1, shape=([7*7*64]), name='const5'), name='b_den5')
L5 = tf.nn.relu(tf.add(tf.matmul(L4, W_den5, name='Den5'), b_den5, name='matmul5'), name='Relu5')

L6 = tf.reshape(L5, [-1, 7, 7, 64], name='L6')

#L7 = tf.layers.conv2d_transpose(L6, 64, 4, strides=2, padding='SAME', name='L7')
W_conv7 = tf.Variable(tf.truncated_normal(([4, 4, 64, 64]), stddev=0.1, name='C_conv7'), name='W_conv7')
b_conv7 = tf.Variable(tf.constant(0.1, shape=([64]), name='const7'), name='b_conv7')
L7 = tf.nn.relu(tf.nn.conv2d_transpose(L6, W_conv7,output_shape=[batch_size,14,14,64], strides=[1, 2, 2, 1], padding='SAME', name='upcon7') + b_conv7, name='Relu7')

#L8 = tf.layers.conv2d_transpose(L7, 64, 4, strides=1, padding='SAME', name='L8')
W_conv8 = tf.Variable(tf.truncated_normal(([4, 4, 64, 64]), stddev=0.1, name='C_conv8'), name='W_conv8')
b_conv8 = tf.Variable(tf.constant(0.1, shape=([64]), name='const8'), name='b_conv8')
L8 = tf.nn.relu(tf.nn.conv2d_transpose(L7, W_conv8,output_shape=[batch_size,14,14,64], strides=[1, 1, 1, 1], padding='SAME', name='upcon8') + b_conv8, name='Relu8')

#L9 = tf.layers.conv2d_transpose(L8, 64, 4, strides=1, padding='SAME', name='L9')
W_conv9 = tf.Variable(tf.truncated_normal(([4, 4, 64, 64]), stddev=0.1, name='C_conv9'), name='W_conv9')
b_conv9 = tf.Variable(tf.constant(0.1, shape=([64]), name='const9'), name='b_conv9')
L9 = tf.nn.relu(tf.nn.conv2d_transpose(L8, W_conv9, output_shape=[batch_size,14,14,64], strides=[1, 1, 1, 1], padding='SAME', name='upcon9') + b_conv9, name='Relu9')

#L10 = tf.layers.conv2d_transpose(L9, 1, 4, strides=2, activation=tf.nn.sigmoid, padding='SAME', name='L10')
W_conv10 = tf.Variable(tf.truncated_normal(([4, 4, 1, 64]), stddev=0.1, name='C_conv10'), name='W_con10')
b_conv10 = tf.Variable(tf.constant(0.1, shape=([1]), name='const10'), name='b_conv10')
L10 = tf.nn.sigmoid(tf.nn.conv2d_transpose(L9, W_conv10,output_shape=[batch_size,28,28,1], strides=[1, 2, 2, 1], padding='SAME', name='upcon10') + b_conv10, name='Relu10')





#return L10


#result = build_CNN_classifier(ori, z)

loss = tf.reduce_mean(tf.square(y - L10))
train_step = tf.train.AdamOptimizer(1e-3, name='adam1').minimize(loss)


def next_batch(num, x, output, noise):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0, len(x))
    np.random.shuffle(idx)
    idx = idx[:num]
    x_shuffle = [x[i] for i in idx]
    output_shuffle = [output[i] for i in idx]
    noise_shuffle = [noise[i] for i in idx]

    return np.asarray(x_shuffle), np.asarray(output_shuffle), np.asarray(noise_shuffle)


saver = tf.train.Saver()
with tf.Session() as sess:
    # 모든 변수들을 초기화한다.
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "d:/asset/extracting_generator_tensorflow/my_test_model6")
    # 10000 Step만큼 최적화를 수행합니다.
    for i in range(100):
        batch = next_batch(100, data_zero, data_main, data_noise_mod)

        # 10 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
        if i % 10 == 0:
            loss_print = loss.eval(feed_dict={ori: batch[0] / 255, y: batch[1] / 255, z: batch[2]})

            print("반복(Epoch): %d, 손실 함수(loss): %f" % (i, loss_print))

        # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
        sess.run(train_step, feed_dict={ori: batch[0] / 255, y: batch[1] / 255, z: batch[2]})
        if i == 99:
            saver.save(sess, 'my_test_model_low_level_tensorflow2')
            #saver.restore(sess, 'd:/asset/extracting_generator_tensorflow/my_test_model_low_level_tensorflow2')
            ##a, b= sess.run(W_conv1.eval())
            #c, d= sess.run(b_conv1.eval())
            #a = W_conv1.eval(session=sess)

            content_feature1 = L1.eval(feed_dict={ori: context_img2})
            content_feature2 = L3.eval(feed_dict={ori: context_img2})

            np.save('content_feature1', content_feature1)
            np.save('content_feature2', content_feature2)

            W_conv1_s = W_conv1.eval(session=sess)
            b_conv1_s = b_conv1.eval(session=sess)

            W_den3_s = W_den3.eval(session=sess)
            b_den3_s = b_den3.eval(session=sess)

            W_den5_s = W_den5.eval(session=sess)
            b_den5_s = b_den5.eval(session=sess)

            W_conv7_s = W_conv7.eval(session=sess)
            b_conv7_s = b_conv7.eval(session=sess)

            W_conv8_s = W_conv8.eval(session=sess)
            b_conv8_s = b_conv8.eval(session=sess)

            W_conv9_s = W_conv9.eval(session=sess)
            b_conv9_s = b_conv9.eval(session=sess)

            W_conv10_s = W_conv10.eval(session=sess)
            b_conv10_s = b_conv10.eval(session=sess)

