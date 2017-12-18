'''
Chenxi Gao

For set up:
need to have annotation.txt file under the current path
to evaluate the accuracy of the model
and return the accuracy
Terminal command example:
python predict.py 'examples/'
or python3 predict.py 'examples/'
special note:
there might be warning messages shown on terminal
but does not affect the result
credit:
http://niektemme.com/
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax_xla.py
'''


from __future__ import print_function
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import glob
import re

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(init_op)
        #getting the model that is created and saved
        saver.restore(sess, "./model2.ckpt")
        #getting first value as prediction
        prediction=tf.argmax(y_conv,1)
        return prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)

#preparing the images
def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    #creates dark canvas of 28x28 pixels as testing images has dark background
    newImage = Image.new('L', (28, 28), (0))

    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        print(nheight)
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on black canvas
    else:
        #if Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on black canvas

    #newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values

    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    #we want the white ones as targeted pixels as the background of test images now is black
    tva = [ x*1.0/255.0 for x in tv]
    return tva

def main(argv):

    #pass the images for preprocessing
    imvalue = imageprepare(argv)
    #predict the number using the model we saved
    predint = predictint(imvalue)
    #getting the correct answers from text file
    return (predint[0]) #first value in list


#return value for sorting the images in path
#because glob.glob does not sort
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#caculate the accuracy in our file
#this step is optional
def getAccuracy(passN):
    sup = []
    predList=[]
    cor=0
    #reading from output file and the annotation to compare
    with open("annotation.txt") as annot_file:
        data1 = annot_file.readlines()
        for line1 in data1:
            supposed=int(line1.split()[1])
            sup.append(supposed)
    with open("predictions.txt") as f:
        data2 = f.readlines()
        for line2 in data2:
            pred=int(line2.split()[1])
            predList.append(pred)

    for i in range(0, len(sup)):
        if sup[i]==predList[i]:
            cor+=1
    return (float(cor)/passN)


if __name__ == "__main__":

    #couting number of images passed to the model for later evaluation
    passN = 0
    #accessing images in the folder
    for filename in sorted(glob.glob(sys.argv[1]+'*.png'),key=numericalSort):
        # print(filename)
        #evaluate current image and return prediction
        pred=main(filename)
        passN = passN+1
        with open("predictions.txt", "a") as f:
            f.write(filename.split('/',1)[1] +'\t'+ str(pred)+'\n')
        #reset after each prediction
        tf.reset_default_graph()
        f.close()
    acc=getAccuracy(passN)
print('Accuracy is: ' + str(acc))
