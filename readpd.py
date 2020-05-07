from tensorflow.python.platform import gfile
from tensorflow import keras
import tensorflow.compat.v1 as tf
import tensorflow

mnist=keras.datasets.mnist
# global_var=tf.global_variables_initializer()
sess = tf.Session()
# f=tf.gfile.GFile('mymodel.pb', 'rb')
# graph_def=tf.GraphDef()
# graph_def.ParseFromString(f.read())
# sess.graph.as_default()
# tf.import_graph_def(graph_def, name='acc/Mean')
with gfile.FastGFile('mymodel.pb', 'rb') as f:
    print(type(f))
    graph_def = tf.GraphDef()
    tmpread=f.read()
    graph_def.ParseFromString(tmpread)
    # print(tmpread)
    sess.graph.as_default()
    x=tf.import_graph_def(graph_def,return_elements=['acc/Mean'], name='')

# global_var=tf.global_variables_initializer()
# print(type(global_var))
# sess.run(global_var)
# sess.run(tf.global_variables_initializer())

# AttributeError: 'NoneType' object has no attribute 'run'
(mnist_x,mnist_y),(_,_)=mnist.load_data()
mnist_x=mnist_x.reshape(mnist_x.shape[0],784)
mnist_y=keras.utils.to_categorical(mnist_y,10)
train_op=tf.GraphDef()
input_x=sess.graph.get_tensor_by_name('x')
input_true=sess.graph.get_tensor_by_name('y_true')
sess.run(train_op, feed_dict={input_x: mnist_x,input_true: mnist_y})
