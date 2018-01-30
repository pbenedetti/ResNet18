import numpy as np
import sys
import os
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

def getBatch(X, Y, batch_size, i):
    start_id = i*batch_size
    end_id = min( (i+1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    batch_y = Y[start_id:end_id]
    return batch_x, batch_y

def getLabelFormat(Y):
	vals = np.unique(np.array(Y))
	sorted(vals)
	hash_val = {}
	for el in vals:
		hash_val[el] = len(hash_val.keys())
	new_Y = []
	for el in Y:
		t = np.zeros(len(vals))
		t[hash_val[el]] = 1.0
		new_Y.append(t)
	return np.array(new_Y)


def ConvBnPool(x,n_filter):

    conv = tf.layers.conv2d( x,
                             filters=n_filter,
                             kernel_size=[7, 7],
                             strides=(2,2),
                             padding="same",
                             activation=tf.nn.relu)

    bn = tf.layers.batch_normalization(conv)

    return bn

def BlockEval( x, drop_rate, n_filter, n_stride, itr,tf_is_traing_pl ):

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

    conv2 = tf.layers.conv2d( relu1,
                              filters=n_filter,
                              kernel_size=[3, 3],
                              padding="same")


    bn2 = tf.layers.batch_normalization( conv2 )

    summ = tf.add( shortcut, bn2 )

    relu2 = tf.nn.relu( summ )

    drop_out = tf.layers.dropout( relu2,
                                 rate=drop_rate,
                                 training=tf_is_traing_pl,
                                 name="drop_layer")

    return drop_out

def ClassificationLayer( x, n_filter ,n_classes):
	conv1 = tf.layers.conv2d( x,
                              filters = n_filter,
                              kernel_size = [1, 1],
                              padding = "same",
                              activation = tf.nn.relu)

	features = tf.reduce_max(conv1, [ 1, 2 ], name="features")

	output = tf.layers.dense( features,
                            units = n_classes,
                            activation = tf.nn.relu,
                            name = "prediction")

	return output,features

def ResNet( x, drop_rate, n_filter,n_classes, height, width, tf_is_traing_pl):

    outCBP = ConvBnPool(x,n_filter)

    block_iter = 2

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

            outDense = BlockEval( aux_input, drop_rate, aux_filter, n_stride, i, tf_is_traing_pl )
            aux_input = outDense

        aux_filter = aux_filter*2

    logits, features = ClassificationLayer( outDense, n_filter, n_classes )

    return logits,features

# MAIN
# Parameters
itr = int( sys.argv[1] )
p_split = 100*float( sys.argv[2] )
batch_size = int( sys.argv[3] )
hm_epochs= int( sys.argv[4] )
n_filter = 32
learning_rate = 0.001
g_path = 'dataset/'

directory = g_path+'ResNet18_%d_%d/'%( p_split, batch_size )
if not os.path.exists(directory):
    os.makedirs(directory)

path_out_model = directory+'modelTT%d/'%itr
if not os.path.exists(path_out_model):
    os.makedirs(path_out_model)

#log file
f = open(directory+"log.txt","a")

#dataset path
var_train_x = g_path + 'VHSR/train_x%d_%d.npy'%(itr,p_split)
var_train_y = g_path+'ground_truth/train_y%d_%d.npy'%(itr,p_split)

var_test_x = g_path+'VHSR/test_x%d_%d.npy'%(itr,p_split)
var_test_y = g_path+'ground_truth/test_y%d_%d.npy'%(itr,p_split)

#load dataset
train_x = np.load(var_train_x)
train_y = np.load(var_train_y)

test_x = np.load(var_test_x)
test_y = np.load(var_test_y)

# Network Parameters
height = train_x.shape[2]
width = height
band = train_x.shape[1]
n_classes = np.bincount( test_y ).shape[0]-1

drop_rate = 0.2 # Dropout, percentage of dropped input units
#tf_is_traing_pl = tf.Variable( True ) #drop out traininv variable to set False when testing
tf_is_traing_pl = tf.placeholder(tf.bool, shape=(), name="is_training")

# It is better to use 2 placeholders, to avoid to load all data into memory,
# and avoid the 2Gb restriction length of a tensor.
x = tf.placeholder("float",[None,height,width,band],name="x")
y = tf.placeholder( tf.float32, [ None, n_classes ], name = "y" )

sess = tf.InteractiveSession()

prediction, feat = ResNet( x, drop_rate, n_filter,n_classes, height, width, tf_is_traing_pl)

tensor1d = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
cost = tf.reduce_mean(tensor1d)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate ).minimize(cost)

correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float64))

tf.global_variables_initializer().run()

#initialise to save model
saver = tf.train.Saver()

#format data
train_y = getLabelFormat(train_y)
test_y = getLabelFormat(test_y)
#(90857,65,65,4)
train_x = np.swapaxes(train_x, 1, 3)
test_x = np.swapaxes(test_x, 1, 3)

iterations = train_x.shape[0] / batch_size

if train_x.shape[0] % batch_size != 0:

    iterations+=1

best_loss = 1
for e in range(hm_epochs):
    #shuffle input
    train_x, train_y = shuffle(train_x, train_y)

    lossfunc = 0
    accuracy_ = 0

    for batch_id in range( iterations ):

        batch_x, batch_y = getBatch(train_x, train_y, batch_size, batch_id )

        acc,_,c = sess.run([accuracy,optimizer,cost], feed_dict={x: batch_x, y:batch_y, tf_is_traing_pl:True })
        accuracy_ += acc
        lossfunc += c

    loss_epoch = float(lossfunc/iterations)
    acc_epoch = float(accuracy_/iterations)
    print "epoch %d Train loss:%f| Accuracy: %f"%(e,loss_epoch,acc_epoch)
    f.write("epoch %d Train loss:%f| Accuracy: %f\n"%(e,loss_epoch,acc_epoch))

    if loss_epoch < best_loss:
        best_loss = loss_epoch
        saver.save(sess,path_out_model+'my_model',global_step=itr)

f.close()
