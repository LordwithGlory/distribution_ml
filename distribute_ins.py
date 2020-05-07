import tensorflow.compat.v1 as tf
# from tensorflow.compat.v1.examples.tutorials.mnist import input_data
import tensorflow
from tensorflow import keras
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

tf.disable_eager_execution()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("job_name", "worker", "启动服务类型，ps或者worker")
tf.app.flags.DEFINE_integer("task_index", 0, "指定是哪一台服务器索引")


def main(argv):

    # 集群描述
    cluster = tf.train.ClusterSpec({
        "ps": ["172.17.0.2:9666"],
        "worker": ["172.17.0.3:9666"]
    })

    # 创建不同的服务
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    else:
        work_device = "/job:worker/task:0/cpu:0"
        with tf.device(tf.train.replica_device_setter(
            worker_device=work_device,
            cluster=cluster
        )):

            # 全局计数器
            global_step = tf.train.get_or_create_global_step()

            # 准备数据
            # mnist=tensorflow.keras.datasets.mnist
            mnist=keras.datasets.mnist
            # mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)

            # 建立数据的占位符
            with tf.variable_scope("data"):
                x = tf.placeholder(tf.float32, [None, 28 * 28], name='x')
                y_true = tf.placeholder(tf.float32, [None, 10], name='y_true')

            # 建立全连接层的神经网络
            with tf.variable_scope("fc_model"):
                # 随机初始化权重和偏重
                weight = tf.Variable(tf.random_normal([28 * 28, 10], mean=0.0, stddev=1.0), name="w")
                bias = tf.Variable(tf.constant(0.0, shape=[10]), name="b")
                # 预测结果
                y_predict = tf.matmul(x, weight) + bias
                y_predict = tf.Variable(y_predict,name="predict")

            # 所有样本损失值的平均值
            with tf.variable_scope("soft_loss"):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

            # 梯度下降
            with tf.variable_scope("optimizer"):
                train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss, global_step=global_step, name="train_op")

            # 计算准确率
            with tf.variable_scope("acc"):
                equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
                accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32),name="accuracy")

        # 创建分布式会话
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir="./temp/ckpt/test",
            master="grpc://172.17.0.3:9666",
            is_chief=(FLAGS.task_index == 0),
            config=tf.ConfigProto(log_device_placement=True),
            hooks=[tf.train.StopAtStepHook(last_step=100)]
        ) as mon_sess:
            while not mon_sess.should_stop():
                # mnist_x, mnist_y = mnist.train.next_batch(4000)
                (mnist_x,mnist_y),(_,_)=mnist.load_data()
                mnist_x=mnist_x.reshape(mnist_x.shape[0],784)
                mnist_y=keras.utils.to_categorical(mnist_y,10)
                mon_sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = graph_util.convert_variables_to_constants(mon_sess,graph_def,output_node_names=["acc/Mean","predict"])
                mf=tf.gfile.GFile("mymodel.pb","wb")
                mf.write(output_graph_def.SerializeToString())
                print("训练第%d步, 准确率为%f" % (global_step.eval(session=mon_sess), mon_sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))
                if global_step.eval(session=mon_sess)==99:
                    # graph_def = tf.get_default_graph().as_graph_def()
                    # output_graph_def = graph_util.convert_variables_to_constants(mon_sess,graph_def,[])
                    # with  tf.gfile.GFile("mymodel.pb","wb") as mf:
                    #     serialized_graph = output_graph_def.SerializeToString()
                    #     mf.write(serialized_graph)
                    # model_f.write(output_graph_def.SerializeToString())
                    save_path="save_models/mymodel"
                    # train_op mf graph_def 3 paras can't be the first para

                    # keras.models.save_model(graph_def, save_path, save_format="tf")
                    break


if __name__ == '__main__':
    tf.app.run()
