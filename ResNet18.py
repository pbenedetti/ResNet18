import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def ConvBnPool(x,n_filter):

    conv = tf.layers.conv2d( x,
                             filters=n_filter,
                             kernel_size=[7, 7],
                             strides=(2,2),
                             padding="same",
                             activation=tf.nn.relu)

    bn = tf.layers.batch_normalization(conv)

    #max_pool = tf.layers.max_pooling2d( bn,
    #                                   pool_size = [3, 3],
    #                                   strides=2 )

    print"\n=======ConvBnPool==========="
    print"input: ",x.get_shape()
    print"conv7: ",conv.get_shape()
    print"bn: ",bn.get_shape()
    #print"max_pool3: ",max_pool.get_shape()
    print"================================="
    print"================================="

    return bn

def BlockEval( x, drop_rate, n_filter, n_stride, itr ):

    #identity shortcut when dimention increase , we apply EQN(2)
    #to match dimentions using a 1x1 convolution
    if n_stride == 2:
        shortcut = tf.layers.conv2d( x,
                            filters=n_filter,
                            kernel_size=[1, 1],
                            padding="valid",
                            strides = ( n_stride, n_stride ))

        shortcut = tf.layers.batch_normalization( shortcut )
        shortcut = tf.nn.relu( shortcut )

        print "padded x",shortcut.get_shape()

    else:
        shortcut = x

    #this tride is the reason why in paper the tensor dimention decrease
    conv1 = tf.layers.conv2d( x,
                            filters=n_filter,
                            kernel_size=[3, 3],
                            padding="same",
                            strides = ( n_stride, n_stride ))

    #we adopt batch normalization right after each convolution and before activation
    bn1 = tf.layers.batch_normalization( conv1 )
    relu1 = tf.nn.relu( bn1 )
    print"conv3x3_bn_relu: ",relu1.get_shape()

    conv2 = tf.layers.conv2d( relu1,
                              filters=n_filter,
                              kernel_size=[3, 3],
                              padding="same")


    bn2 = tf.layers.batch_normalization( conv2 )
    print"conv3x3_bn: ",bn2.get_shape()

    summ = tf.add( shortcut, bn2 )

    relu2 = tf.nn.relu( summ )

    drop_out = tf.layers.dropout( relu2,
                                 rate=drop_rate,
                                 noise_shape=None,
                                 seed=None,
                                 training=False,
                                 name=None)

    print"\n=======Block_%d_filter(%d)==========="%(itr,n_filter)
    print"input: ",x.get_shape()
    print "stride: ",n_stride
    print "shortcut: ",shortcut.get_shape()
    print"conv3x3_bn_relu: ",relu1.get_shape()
    print"conv3x3_bn: ",bn2.get_shape()
    print"summ: ",summ.get_shape()
    print"relu: ",relu2.get_shape()
    print"drop_out: ",drop_out.get_shape()
    print"================================="
    print"================================="

    return drop_out

def ClassificationLayer( x, n_filter ):

    conv1 = tf.layers.conv2d( x,
                              filters = n_filter,
                              kernel_size = [1, 1],
                              padding = "same",
                              activation = tf.nn.relu)

    pool_flat = tf.reduce_mean(conv1, [ 1, 2 ])

    features = tf.layers.dense( pool_flat,
                            units = n_filter,
                            activation = tf.nn.relu)

    print"\n=======ClassificationLayer==========="
    print"input: ",x.get_shape()
    print"conv1: ",conv1.get_shape()
    print"pool: ",pool_flat.get_shape()
    print"features: ",features.get_shape()
    print"================================="
    print"================================="

    return features

def ResNet( x, drop_rate, n_filter, height, width):

    input_layer = tf.reshape(x, [-1, height, width, 1])

    outCBP = ConvBnPool(input_layer,n_filter)

    print"x:", x.get_shape()
    print"input_layer:", input_layer.get_shape()

    block_iter = 2
    print "n_filter: %d"%n_filter
    print"outCBP:", outCBP.get_shape()

    aux_input = outCBP
    aux_filter = n_filter
    n_stride = 1
    for rep_block in range(4):
        for i in range(block_iter):

            #add a 2 stride when entering new filter block
            if ( i == 0 ) & ( rep_block > 0 ):
                n_stride = 2

            else:
                n_stride = 1

            outDense = BlockEval( aux_input, drop_rate, aux_filter, n_stride, i )
            aux_input = outDense
            #print"aux_input:", aux_input.get_shape()

        aux_filter = aux_filter*2
        print "n_filter: %d"%aux_filter

    features = ClassificationLayer( outDense, n_filter )

    logits = tf.layers.dense(features, units=10)

    return logits,features

def train( x, drop_rate, n_filter, n_classes, height, width, hm_epochs ):

    sess = tf.InteractiveSession()

    prediction, feat = ResNet( x, drop_rate, n_filter, height, width )

    print"\n=======TRAIN==========="
    print"prediction: ", prediction.get_shape()
    print"feat: ", feat.get_shape()

    tensor1d = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
    cost = tf.reduce_mean(tensor1d)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    tf.global_variables_initializer().run()

    for epoch in range(hm_epochs):

        epoch_loss=0

        #for _ in range(int(mnist.train.num_examples/batch_size)):
        for _ in range(2):
            batch_x,batch_y = mnist.train.next_batch(batch_size)

            print "batch_n: %d"%_
            print "x_shape:",batch_x.shape

            _, c=sess.run([optimizer, cost],feed_dict={x: batch_x, y: batch_y, drop_rate:0.8})
            epoch_loss+=c

        print'Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss

    correct = tf.equal(tf.argmax(prediction, 1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))

    print'accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

height = 28
width = 28
batch_size = 128
n_classes = 10
n_filter = 64
hm_epochs = 5

drop_rate = tf.placeholder( tf.float32, [], name = "drop_rate" )
x = tf.placeholder( tf.float32, [ None, height*width ], name = "x" )
y = tf.placeholder( tf.float32, [ None, n_classes ], name = "y" )

train( x, drop_rate, n_filter, n_classes, height, width, hm_epochs )
