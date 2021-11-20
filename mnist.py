import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

n_input = 784
input_width = 28
input_height = 28
input_channels = 1

n_conv1 = 32
n_conv2 = 64
conv1_k = 5
conv2_k = 5

n_hidden = 1024
n_out = 10

pooling_window_size = 2

conv2_out_width = input_width//(pooling_window_size*pooling_window_size)
conv2_out_height = input_height//(pooling_window_size*pooling_window_size) 
dense_layer_input = conv2_out_width*conv2_out_height * n_conv2

weights = {
    'c1': tf.Variable(tf.random_normal([conv1_k,conv1_k,input_channels,n_conv1])),
    'c2': tf.Variable(tf.random_normal([conv2_k,conv2_k, n_conv1,n_conv2])),
    'd1': tf.Variable(tf.random_normal([dense_layer_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_out]))
}

biases = {
    'c1': tf.Variable(tf.random_normal([n_conv1])),
    'c2': tf.Variable(tf.random_normal([n_conv2])),
    'd1': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_out]))
}

def conv(x, weights, bias, stride = 1):
    conv_out = tf.nn.conv2d(x, weights, padding="SAME", strides=[1,stride,stride,1])
    out = tf.nn.bias_add(conv_out, bias)
    out = tf.nn.relu(out)
    return out

def maxpooling(x, k = 2):
    return tf.nn.max_pool(x, padding="SAME", ksize=[1, k, k, 1], strides = [1, k, k, 1])

def cnn(x, weights, biases):
    x = tf.reshape(x, shape=[-1,input_width, input_height, input_channels ])
    conv1 = conv(x, weights["c1"], biases['c1'])
    conv1 = maxpooling(conv1, k = pooling_window_size)
    
    conv2 = conv(conv1, weights["c2"], biases['c2'])
    conv2 = maxpooling(conv2, k = pooling_window_size)
    
    hidden_input = tf.reshape(conv2, shape=[-1, dense_layer_input])
    hidden_output = tf.add(tf.matmul(hidden_input, weights['d1']), biases['d1'])
    hidden_output = tf.nn.relu(hidden_output)
    
    in_output_layer = tf.add(tf.matmul(hidden_output, weights['out']), biases['out'])
    output = in_output_layer
    return output
    
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder(tf.int32, [None, n_out])

pred = cnn(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
optimize_step = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100
for i in range(25):
    num_batches = int(mnist.train.num_examples/batch_size)
    total_cost = 0
    for j in range(num_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size) 
        c, _ = sess.run([cost, optimize_step], feed_dict={x:batch_x, y:batch_y})
        total_cost += c
    print(total_cost)
    
predictions = tf.argmax(pred, axis = 1)
actual_labels = tf.argmax(y, axis = 1)
correct_predictions = tf.equal(predictions,actual_labels)

correct_preds = sess.run(correct_predictions, feed_dict={x:mnist.test.images, y:mnist.test.labels} )
(correct_preds.sum()/10000)*100