#!/usr/bin/python3

#Source: https://www.tensorflow.org/versions/r1.2/get_started/mnist/pros

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data #handwritten digits dataset

import time

print("#####==========================####")

######### Load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

dummy = input("Press ENTER to continue")
print("#####==========================####")

print('Total Images in Training Dataset = ' + repr(mnist.train.images.shape[0]))
dim = int(np.sqrt(mnist.train.images.shape[1]))
print("Image size = "+repr(dim)+"x"+repr(dim))

print("")
print('Total Images in Testing Dataset = ' + repr(mnist.test.images.shape[0]))

nClasses = mnist.train.labels.shape[1]
print("")
print("Total labels in Training Dataset = " + repr(nClasses))

print("")
dummy = input("Press ENTER to continue")
print("#####==========================####")

index = 22500
label = mnist.train.labels[index,:].argmax(axis=0)
image = mnist.train.images[index,:].reshape([dim,dim])
plt.figure(1)
plt.subplot(121)
plt.title('Example # in Train set: %d \n Label: %d' % (index, label))
plt.imshow(image, cmap=plt.get_cmap('gray_r'))

index = 5371
label = mnist.test.labels[index,:].argmax(axis=0)
image = mnist.test.images[index,:].reshape([dim,dim])

plt.subplot(122)
plt.title('Example # in Test set: %d \n  Label: %d' % (index, label))
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show(block=False)


print("")
dummy = input("Press ENTER to continue")
print("#####==========================####")

### Input Layer

x = tf.placeholder(tf.float32, shape=[None, dim*dim]) #feed as many images as we want
y_ = tf.placeholder(tf.float32, shape=[None, nClasses]) #feed as many labels (from nClasses) as we want

### Black-&-White images ==> only 1 channel
d_image = 1 #1 channel
x_image = tf.reshape(x, [-1, dim, dim, d_image]) #-1 ==> Tensorflow figures out the correct value based on shape of x!

#Needed format is: [batch_size, width, height, channels/depth]
print("")
print("Shape of x_image:")
print(x_image.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## 1st Conv Layer
nFilters_1 = 32
shape_conv1 = [5, 5, 1, nFilters_1] #32 filters of 5*5*1 shape
### Random initialization of weights of 1st Conv layer
W_conv1 = tf.Variable(tf.truncated_normal(shape_conv1, stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[nFilters_1])) #constant initialization; 1 bias per filter. NOTE: [] for nFilters

op_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
#padding="SAME" is the basic option. With this, for conv2D, Tensorflow will figure out correct padding such that spatial dimensions of input output remain the same
#shape of strides MUST match shape of x_image
print("")
print("Shape of op_conv1:")
print(op_conv1.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## 1st ReLU Layer
op_relu1 = tf.nn.relu(op_conv1)
### we can combine the 2 layers as tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1): Nested functions
print("")
print("Shape of op_relu1:")
print(op_relu1.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## 1st MaxPool Layer
### Recall: no parameters for MaxPool layer :-) ==> so no need for any initialization
op_maxpool1 = tf.nn.max_pool(op_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
#max-pooling on width*height dimensions at rate 1/2; padding="SAME" means basic option
print("")
print("Shape of op_maxpool1:")
print(op_maxpool1.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## 2nd Conv Layer
nFilters_2 = 64
shape_conv2 = [5, 5, 32, nFilters_2] #64 filters of 5*5*32 shape. Why depth = 32??
### Random initialization of weights of 1st Conv layer
W_conv2 = tf.Variable(tf.truncated_normal(shape_conv2, stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[nFilters_2])) #constant initialization; 1 bias per filter. NOTE: [] for nFilters_2

op_conv2 = tf.nn.conv2d(op_maxpool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
print("")
print("Shape of op_conv2:")
print(op_conv2.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## 2nd ReLU Layer
op_relu2 = tf.nn.relu(op_conv2)
print("")
print("Shape of op_relu2:")
print(op_relu2.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## 2nd MaxPool Layer
op_maxpool2 = tf.nn.max_pool(op_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME") #max-pooling on width*height dimensions at rate 1/2
print("")
print("Shape of op_maxpool2:")
print(op_maxpool2.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## Fully Connected Layer
nNeurons = 1024
shape_fc1 = [7*7*64, nNeurons] #1024 neurons processing volume of shape 7*7*64. Why 7*7*64?
W_fc1 = tf.Variable(tf.truncated_normal(shape_fc1, stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[nNeurons])) #constant initialization; 1 bias per filter. NOTE: [] for nNeurons

op_maxpool2_flat = tf.reshape(op_maxpool2, [-1, shape_fc1[0]]) #["appropriate size", 7*7*64]
print("")
print("Shape of op_maxpool2_flat:")
print(op_maxpool2_flat.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

op_fc1 = tf.matmul(op_maxpool2_flat, W_fc1) + b_fc1
print("")
print("Shape of op_fc1:")
print(op_fc1.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## 3rd ReLU Layer
op_relu3 = tf.nn.relu(op_fc1)
print("")
print("Shape of op_relu3:")
print(op_relu3.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## Dropout Layer
keep_prob = tf.placeholder(tf.float32)  #just a scalar
op_relu3_dropout = tf.nn.dropout(op_relu3, keep_prob)
print("")
print("Shape of op_relu3_dropout:")
print(op_relu3_dropout.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")

############## Output (Fully Connected) Layer
W_fc2 = tf.Variable(tf.truncated_normal([nNeurons, nClasses], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[nClasses])) #constant initialization; 1 bias per filter. NOTE: [] for nClasses

y_network = tf.matmul(op_relu3_dropout, W_fc2) + b_fc2
print("")
print("Shape of y_network:")
print(y_network.get_shape().as_list())

print("")
dummy = input("Press ENTER to continue")


############## Netwotk Definition Complete!!    [Q: Why haven't we used the softmax layer? See below.]

############## Define loss for one batch of training examples

loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_network))
# tf.nn.softmax_cross_entropy_with_logits first convert the "logits" argument to probabilities internally
# ==> no need to explicitly define a final softmax layer

############## Define loss optimization operation for one batch of training examples

train_operation_one_batch = tf.train.AdamOptimizer(1e-4).minimize(loss_cross_entropy)

############## Calculate prediction accuracy for one batch of examples

predicted_label = tf.argmax(y_network, 1)
true_label = tf.argmax(y_, 1)
correct_prediction = tf.equal(predicted_label, true_label) #NOTE: argmax on y_network means predicted class with "MAX" probability
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #NOTE: tf.cast just converts BOOLs (0/1) in previous step to FLOATs (0/1) so average be directly calcualted


############## Train the CNN [finally!]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #Necessary to initialize all Tensorflow variables at the beginning

    for i in range(1000): #No of steps for which we'll run the training
        batch = mnist.train.next_batch(50) #Next batch of "50" examples (image, label) pairs

        if i == 1:
            print(batch[0].shape) #images
            print(batch[1].shape) #labels
            print("")
            dummy = input("Press ENTER to continue")

        if i % 100 == 0: #Print the progress of training every 100 batches
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) #dropout irrelevant for inference
            #above command == evaluate "accuracy" node of the computationa
            print('Training batch = %d, accuracy of the training batch = %g %%' % (i, 100*train_accuracy))

        train_operation_one_batch.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) #dropout probability of 0.5
        #Recall: x & y_ were just placeholders; batch[0]/batch[0] are actual data tensors

    print("")
    dummy = input("Press ENTER to continue")

    ############## Use the trained CNN to make predictions on the Test set
    index = 5371
    print("For test image # " + repr(index) + ":")
    label_test1 = predicted_label.eval(feed_dict={x: mnist.test.images[index,:].reshape(1, dim*dim), y_: mnist.test.labels[index,:].reshape(1, nClasses), keep_prob: 1.0})
    print("Predicted label = "+repr(label_test1[0])+",   True label = "+repr(mnist.test.labels[index,:].argmax(axis=0)))

    print("")
    dummy = input("Press ENTER to continue")

    index = 2258
    print("For test image # "+repr(index)+":")
    label_test2 = predicted_label.eval(feed_dict={x: mnist.test.images[index,:].reshape(1, dim*dim), y_: mnist.test.labels[index,:].reshape(1, nClasses), keep_prob: 1.0})
    print("Predicted label = "+repr(label_test2[0])+",   True label = "+repr(mnist.test.labels[index,:].argmax(axis=0)))

    print("")
    dummy = input("Press ENTER to continue")

    index = 7777
    print("For test image # " + repr(index) + ":")
    label_test3 = predicted_label.eval(feed_dict={x: mnist.test.images[index,:].reshape(1, dim*dim), y_: mnist.test.labels[index,:].reshape(1, nClasses), keep_prob: 1.0})
    print("Predicted label = "+repr(label_test3[0])+",   Predicted label = "+repr(mnist.test.labels[index,:].argmax(axis=0)))

    print("")
    dummy = input("Press ENTER to continue")

    index = 2258
    print("For test image # " + repr(index) + ":")
    label_test4 = predicted_label.eval(feed_dict={x: mnist.test.images[index,:].reshape(1, dim*dim), y_: mnist.test.labels[index,:].reshape(1, nClasses), keep_prob: 1.0})
    print("Predicted label = "+repr(label_test4[0])+",   True label = "+repr(mnist.test.labels[index,:].argmax(axis=0)))

    print("")
    dummy = input("Press ENTER to continue")

    print("test accuracy = " + repr(100*accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))+ " %%")
    #NOTE: entire Test set passed to accuracy.eval() in 1 go




















