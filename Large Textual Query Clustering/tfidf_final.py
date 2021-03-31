#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 11:34:09 2021

@author: sde
"""

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

num_clusters = 12
max_iterations = 200
num_components = 3

path = "data_1/train_set.txt"
path_test = "data_1/public_test_set.txt"

file = open(path,'r')
text = file.read()
file.close()
# preprocessing
modtext = text.split('\n')[1:-1]

file = open(path_test,'r')
text = file.read()
file.close()
testtextn = text.split('\n')[1:-1]

tfidf = TfidfVectorizer(use_idf=True,smooth_idf = True)
x = tfidf.fit_transform(modtext)
xtest = tfidf.fit_transform(testtextn)

reduced_data = TruncatedSVD(n_components=num_components).fit_transform(x) #n_iter=100

# calculate distortion for a range of number of cluster
distortions = []
for i in range(4, 20):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=1, max_iter=200,
        tol=1e-04, random_state=0
    )
    km.fit(x)
    distortions.append(km.inertia_)

# plot
plt.plot(range(4, 20), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# Predict and save results
labels = KMeans(n_clusters = 12).fit_predict(x)
labels_color_map = {
    0: '#E81010', 1: '#150E0E', 2: '#F1960A', 3: '#F1EA0A', 4: '#C0F10A',
    5: '#A90AF1', 6: '#F10A2B', 7: '#0A22F1', 8: '#0ADCF1', 9: '#135926',
    10:'#292C2B',11:'#6C6EB6'}
fig, ax = plt.subplots()
for index, instance in enumerate(reduced_data):
    #print(instance, index, labels[index])
    print(index)
    pca_comp_1, pca_comp_2, pca_comp_3 = reduced_data[index]
    color = labels_color_map[labels[index]]
    try:
        ax.scatter(pca_comp_1, pca_comp_2,pca_comp_3,c = color)
    except Exception() as e:
        print(e)
plt.show()


