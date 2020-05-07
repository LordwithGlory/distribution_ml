import tensorflow
import tensorflow.compat.v1 as tf

export_path="save_models/1588688373/"
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess,["server"],export_path)
    graph=tf.get_default_graph()
    feed_dict={"input_ids_1:0":[feature.input_ids],"input_mask_1:0": [feature.input_mask],"segment_ids_1:0": [feature.segment_ids]}
    sess.run('loss/pred_prob:0',feed_dict=feed_dict)