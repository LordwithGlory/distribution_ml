import tensorflow as tf
import os
from tensorflow import keras
import tensorflow_datasets as tfds
import json
import socket
import time

def resize_data(image_data):
    return image_data.reshape(-1,28*28)/255.0

# 带有默认参数的放在 不带参数的后面
def set_config(port,ps_list=[],worker_list=[],method_num=0):
    skt=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    skt.connect(('8.8.8.8',80))
    ip=skt.getsockname()+":"+port
    role="ps"
    taskid=worker_list.index(ip)
    if ip in worker_list:
        role="worker"
    if method_num!=0:
        return
    os.environ["TF_CONFIG"]=json.dumps({
        "cluster":{
            "worker": str(worker_list),
            "ps": str(ps_list)
        },
        "task":{"type": role,"index":taskid}
    })

def load_dir(dir_path):
    dirs=os.listdir(dir_path)
    model_dir=""
    dir_time=time.time()
    for onedir in dirs:
        filepath=dir_path+'/'+onedir
        # filepath=unicode(filepath,'utf-8')
        t=os.path.getctime(filepath)
        if t<dir_time:
            model_dir=filepath
            dir_time=t
    return model_dir

def set_own_config():
    ps=[""]
    worker=[""]
    port=9666
    return ps,worker,str(port)

def complie_model(model):
    model.complie(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
    return model

strategy=tf.distribute.experimental.MultiWorkerMirroredStrategy()
ps,worker,port=set_own_config()
set_config(port,ps,worker)
models_dir="./save_models"
with strategy.scope():
    model=tf.keras.experimental.load_from_saved_model(load_dir(models_dir))
    model=complie_model(model)
    (train_data,train_lable),(test_data,test_label)=keras.datasets.mnist.load_data()
    train_data=resize_data(train_data)
    test_data=resize_data(test_data)
    model.fit(train_data,train_lable,epochs=10)
save_model_path=models_dir+"/{}".format(int(time.time()))
tf.keras.experimental.export_saved_model(model,save_model_path)