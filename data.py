#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sa import find_closest_at
from tqdm import tqdm

def show_dsa(layer_at, pred, layer_at_inject, pred_inject, name = ""):
    class_matrix = {}
    all_idx = []
    right_idx = []
    wrong_idx = []
    for i, label in enumerate(pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)
    
    for i, label_pred in enumerate(pred_inject):
        label = pred[i % len(pred)]
        if label == label_pred:
            right_idx.append(i)
        else:
            wrong_idx.append(i)

    dsa_right = []
    dsa_wrong = []

    for i, at in enumerate(tqdm(layer_at_inject[right_idx])):
        label = pred_inject[i]
        a_dist, a_dot = find_closest_at(at, layer_at[class_matrix[label]])
        b_dist, _ = find_closest_at(
            a_dot, layer_at[list(set(all_idx) - set(class_matrix[label]))]
        )
        dsa_right.append(a_dist / b_dist) 
        
    for i, at in enumerate(tqdm(layer_at_inject[wrong_idx])):
        label = pred_inject[i]
        a_dist, a_dot = find_closest_at(at, layer_at[class_matrix[label]])
        b_dist, _ = find_closest_at(
            a_dot, layer_at[list(set(all_idx) - set(class_matrix[label]))]
        )
        dsa_wrong.append(a_dist / b_dist) 
    
    ''' 
    bincount_right = []   
    bincount_wrong = []  
    dsa_upper_right = min(np.mean(dsa_right), 9.9) + 0.1
    dsa_upper_wrong = min(np.mean(dsa_wrong), 9.9) + 0.1
    dsa_upper = max(dsa_upper_right, dsa_upper_wrong)
    dsa_lower = min(np.min(dsa_right), np.min(dsa_wrong))
    buckets_right = np.digitize(dsa_right, np.linspace(dsa_lower, dsa_upper, 1000))
    buckets_wrong = np.digitize(dsa_wrong, np.linspace(dsa_lower, dsa_upper, 1000))
    
    for i in range(1000):
        bincount_right.append(np.sum(buckets_right==i))
        bincount_wrong.append(np.sum(buckets_wrong==i))
    
    plt.plot(np.linspace(dsa_lower, dsa_upper, 1000), bincount_right, label="right")
    plt.plot(np.linspace(dsa_lower, dsa_upper, 1000), bincount_wrong, label="wrong")
    plt.legend()
    plt.show()  
    '''
    
    return dsa_right, dsa_wrong


layers = ["activation1", "activation2", "activation3", "activation4", "activation5", "pred"]
probs = ["0.025", "0.05", "0.075", "0.1", "0.125"]
mask = ["activation4"]
dsas = []
zeros = []
ones = []

for prob in probs:
    if prob in mask:
        dsas.append([])
        continue
    dsa = np.load("test/{}/activation5_dsa_wrong.npy".format(prob))
    print(prob)
    print("max: {}".format(np.max(dsa)))
    print("avg: {}".format(np.mean(dsa)))
    dsas.append(dsa)
'''
for layer in layers[:-1]:
    if layer in mask:
        dsas.append([])
        continue
    dsa = np.load("test/0.125/{}_dsa_wrong.npy".format(layer))
    dsas.append(dsa)
    print(layer)
    print("max: {}".format(np.max(dsa)))
    print("avg: {}".format(np.mean(dsa)))
'''
plt.boxplot(dsas)
plt.show()
'''
#0.025->0.90945
#0.05->0.90946
#0.075->0.68213
#0.1->0.7763
#0.125->0.78009
normal = "normal"
prob = "0.125"
layer = layers[4]
at_normal = np.load("test/{}/{}.npy".format(normal, layer))
pred_normal = np.load("test/{}/pred.npy".format(normal))
at_inject = np.load("test/{}/{}.npy".format(prob, layer))
pred_inject = np.load("test/{}/pred.npy".format(prob))
dsa_right, dsa_wrong = show_dsa(at_normal, pred_normal, at_inject, pred_inject)
np.save("test/{}/{}_dsa_right.npy".format(prob, layer), dsa_right)
np.save("test/{}/{}_dsa_wrong.npy".format(prob, layer), dsa_wrong)
'''
