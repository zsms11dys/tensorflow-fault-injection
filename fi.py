#!/usr/bin/python

from __future__ import print_function
import sys
import imp
import os
from tensorflow.python.platform import gfile

import numpy as np
import tensorflow as tf
import TensorFI as ti
import matplotlib.pyplot as plt
from sa import find_closest_at
from tqdm import tqdm

def show_dsa(layer_at, pred, layer_at_inject, pred_inject, name = ""):
    class_matrix = {}
    all_idx = []
    for i, label in enumerate(pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    dsa = []

    print("Fetching DSA for " + name)
    for i, at in enumerate(tqdm(layer_at_inject)):
        label = pred_inject[i]
        a_dist, a_dot = find_closest_at(at, layer_at[class_matrix[label]])
        b_dist, _ = find_closest_at(
            a_dot, layer_at[list(set(all_idx) - set(class_matrix[label]))]
        )
        dsa.append(a_dist / b_dist) 
     
    bincount = []    
    dsa_upper = min(np.mean(dsa), 9.9) + 0.1
    buckets = np.digitize(dsa, np.linspace(np.min(dsa), dsa_upper, 1000))
    target_cov = len(list(set(buckets))) / float(1000) * 100

    print("DSA min: " + str(np.min(dsa)))
    print("DSA max: " + str(np.max(dsa)))
    print("DSA avg: " + str(np.mean(dsa)))
    print()
    
    for i in range(1000):
        bincount.append(np.sum(buckets==i))
    
    plt.plot(np.linspace(np.min(dsa), dsa_upper, 1000), bincount)
    plt.show()  
    
def paintbar(pred):
    count = []          
    for i in range(10):
        count.append(list(pred).count(i))
    plt.bar(range(10), count)
    plt.show()
    
def getacc(pred, label):
    correct_prediction = tf.equal(pred, label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.eval()
    
def paintNodeName(graph):
    [print(n.name) for n in graph.as_graph_def().node]
    
def savegraph(graph):
    writer = tf.summary.FileWriter("model/graph", graph = graph)
    writer.close()
   
def paintOps(graph):
    ops = set()
    for op in graph.get_operations():
        ops.add(op.type)
    print(ops)

with tf.Session() as sess:
    filename = "model/model_mnist_Xle.pb"
    with gfile.FastGFile(filename ,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        
    graph = tf.get_default_graph()
    
    #paintNodeName(graph)
    paintOps(graph)
    #savegraph(graph)
    
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    CLIP_MAX = 0.5
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
    
    dense1 = graph.get_tensor_by_name("dense_1/BiasAdd:0")
    dense2 = graph.get_tensor_by_name("dense_2/BiasAdd:0")
    dense3 = graph.get_tensor_by_name("dense_3/BiasAdd:0")
    activation1 = graph.get_tensor_by_name("activation_1/Relu:0")
    activation2 = graph.get_tensor_by_name("activation_2/Relu:0")
    activation3 = graph.get_tensor_by_name("activation_3/Relu:0")
    activation4 = graph.get_tensor_by_name("activation_4/Relu:0")
    preds = graph.get_tensor_by_name("activation_5/Softmax:0")
    
    if "le" in filename:
        image = x_test.reshape(-1, 28, 28, 1)
        conv1 = graph.get_tensor_by_name("conv2d_1/BiasAdd:0")
        maxpool1 = graph.get_tensor_by_name("max_pooling2d_1/MaxPool:0")
        conv2 = graph.get_tensor_by_name("conv2d_2/BiasAdd:0")
        maxpool2 = graph.get_tensor_by_name("max_pooling2d_2/MaxPool:0")
        
        #'''
        c1, p1, at1, c2, p2, at2, d1, at3, d2, at4, d3, res = \
            sess.run([conv1, maxpool1, activation1, conv2, maxpool2,
                      activation2, dense1, activation3, dense2, activation4, dense3, preds],
                     feed_dict={"conv2d_1_input:0": image}) 
               
        np.save("test/normal/conv1.npy", c1)
        np.save("test/normal/maxpool1.npy", p1)  
        np.save("test/normal/conv2.npy", c2)
        np.save("test/normal/maxpool2.npy", p2)
        #'''
        
    else:
        image = x_test.reshape(-1, 784)
        dense4 = graph.get_tensor_by_name("dense_4/BiasAdd:0")
        dense5 = graph.get_tensor_by_name("dense_5/BiasAdd:0")
 
        #'''
        at1, at2, at3, at4, res, d1, d2, d3, d4, d5 = \
            sess.run([activation1, activation2, activation3, activation4, preds,
                      dense1, dense2, dense3, dense4, dense5],
                     feed_dict={"dense_1_input:0": image}) 
              
        np.save("test/normal/dense4.npy", d4)
        np.save("test/normal/dense5.npy", d5) 
    
    np.save("test/normal/activation1.npy", at1)
    np.save("test/normal/activation2.npy", at2)
    np.save("test/normal/activation3.npy", at3)
    np.save("test/normal/activation4.npy", at4)
    np.save("test/normal/activation5.npy", res)
    np.save("test/normal/dense1.npy", d1)
    np.save("test/normal/dense2.npy", d2)
    np.save("test/normal/dense3.npy", d3)
         
    pred = np.argmax(res, axis = 1)
    print("no injection acc:{}".format(getacc(pred, y_test)))       
    np.save("test/normal/pred.npy", pred)
        #'''
        
    #savegraph(graph)
      
    for i in range(1):
        print(i)
        fi = ti.TensorFI(sess, logLevel = 100, name = "mnist")
        fi.turnOnInjections()   
        
        if "le" in filename:
             c1, p1, at1, c2, p2, at2, d1, at3, d2, at4, d3, res = \
                sess.run([conv1, maxpool1, activation1, conv2, maxpool2,
                      activation2, dense1, activation3, dense2, 
                      activation4, dense3, preds],
                     feed_dict={"conv2d_1_input:0": image})  
        else:
            at1, at2, at3, at4, res, d1, d2, d3, d4, d5 = \
                sess.run([activation1, activation2, activation3, activation4, preds,
                      dense1, dense2, dense3, dense4, dense5],
                     feed_dict={"dense_1_input:0": image})        
        fi.turnOffInjections()    
        
        pred = np.argmax(res, axis = 1) 
        print("injection acc:{}".format(getacc(pred, y_test)))
'''        
        if i == 0:
            at1_pre = at1
            at2_pre = at2
            at3_pre = at3
            at4_pre = at4
            at5_pre = at5
            all_pred = pred
            dense1_pre = d1
            dense2_pre = d2
            dense3_pre = d3
            dense4_pre = d4
            dense5_pre = d5
            
        else:
            at1_pre = np.concatenate((at1_pre, at1))
            at2_pre = np.concatenate((at2_pre, at2))
            at3_pre = np.concatenate((at3_pre, at3))
            at4_pre = np.concatenate((at4_pre, at4))
            at5_pre = np.concatenate((at5_pre, at5))
            all_pred = np.concatenate((all_pred, pred))
            dense1_pre = np.concatenate((dense1_pre, d1))
            dense2_pre = np.concatenate((dense2_pre, d2))
            dense3_pre = np.concatenate((dense3_pre, d3))
            dense4_pre = np.concatenate((dense4_pre, d4))
            dense5_pre = np.concatenate((dense5_pre, d5))
        
    print(at1_pre.shape)
    print(at2_pre.shape)
    print(at3_pre.shape)
    print(at4_pre.shape)
    print(at5_pre.shape)
    print(all_pred.shape)
    print(dense1_pre.shape)
    print(dense2_pre.shape)
    print(dense3_pre.shape)
    print(dense4_pre.shape)
    print(dense5_pre.shape)
    np.save("test/0.15/activation1.npy", at1_pre)
    np.save("test/0.15/activation2.npy", at2_pre)
    np.save("test/0.15/activation3.npy", at3_pre)
    np.save("test/0.15/activation4.npy", at4_pre)
    np.save("test/0.15/activation5.npy", at5_pre)
    np.save("test/0.15/pred.npy", all_pred)
    np.save("test/0.15/dense1.npy", dense1_pre)
    np.save("test/0.15/dense2.npy", dense2_pre)
    np.save("test/0.15/dense3.npy", dense3_pre)
    np.save("test/0.15/dense4.npy", dense4_pre)
    np.save("test/0.15/dense5.npy", dense5_pre)
        
    #show_dsa(activation1_at, pred, activation1_at_in, pred_in, "activation_1/Relu")
    
    #show_dsa(activation2_at, pred, activation2_at_in, pred_in, "activation_2/Relu")
'''  
'''
DNN
dense_1/BiasAdd
activation_1/Relu
dense_2/BiasAdd
activation_2/Relu
dense_3/BiasAdd
activation_3/Relu
dense_4/BiasAdd
activation_4/Relu
dense_5/BiasAdd
activation_5/Softmax

CNN
conv2d_1/BiasAdd
activation_1/Relu
max_pooling2d_1/MaxPool
conv2d_2/BiasAdd
activation_2/Relu
max_pooling2d_2/MaxPool
dense_1/BiasAdd
activation_3/Relu
dense_2/BiasAdd
activation_4/Relu
dense_3/BiasAdd
activation_5/Softmax


'''
