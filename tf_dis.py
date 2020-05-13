import tensorflow as tf
import os
from tensorflow import keras
import json
import socket
import time

def resize_data(image_data):
    return image_data.reshape(-1,28*28)/255.0

# 带有默认参数的放在 不带参数的后面
def set_config(port,ps_list=[],worker_list=[],method_num=0):
    skt=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    skt.connect(('8.8.8.8', 80))
    ip=skt.getsockname()[0]+":"+port
    role="ps"
    taskid=worker_list.index(ip)
    if ip in worker_list:
        role="worker"
    if method_num!=0:
        return
    os.environ["TF_CONFIG"]=json.dumps({
        "cluster":{
            "worker": worker_list,
            "ps": ps_list
        },
        "task":{"type": role,"index":taskid}
    })

def load_dir(dir_path):
    dirs=os.listdir(dir_path)
    model_dir=""
    dir_time=None
    for onedir in dirs:
        filepath=dir_path+'/'+onedir
        print(filepath)
        # filepath=unicode(filepath,'utf-8')
        t=os.path.getctime(filepath)
        if dir_time==None or t>dir_time:
            model_dir=filepath
            dir_time=t
    return model_dir+'/'

def set_own_config():
    ps=["172.17.0.2:9666"]
    worker=["172.17.0.3:9666"]
    port=9666
    return ps,worker,str(port)

def complie_model(model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
    return model

strategy=tf.distribute.experimental.MultiWorkerMirroredStrategy()
ps,worker,port=set_own_config()
set_config(port,ps,worker)
models_dir="./save_models"
with strategy.scope():
    modelpath=load_dir(models_dir)+'mymodel.h5'
    print(modelpath)
    model=keras.models.load_model(modelpath)
    # model=tf.saved_model.load(modelpath,tags='train')
    model=complie_model(model)
    (train_data,train_lable),(test_data,test_label)=keras.datasets.mnist.load_data()
    train_data=resize_data(train_data)
    test_data=resize_data(test_data)
    model.fit(train_data,train_lable,epochs=10)
save_model_path=models_dir+"/{}".format(int(time.time()))
tf.saved_model.save(model,save_model_path)
model.save(save_model_path+'/'+"mymodel.h5")
