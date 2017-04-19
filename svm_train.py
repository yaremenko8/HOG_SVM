from sklearn import svm
from hog import extract_hog
from multiprocessing.pool import ThreadPool, Pool
import numpy as np
import random
import time
import pickle
from beep import beep

path    = "//home/grigory/train/"
samples_from_class = 195
tests_from_class   = 25

bound = lambda a, c: a if a < c else c

def extract_class_samples(cls):
    sfc = bound(samples_from_class, len(cls) - tests_from_class)
    tr_labels  = [cls[0][1]] * sfc
    temp1 = random.sample(cls, sfc)
    tr_samples = list(map(lambda x: extract_hog(path + x[0]), temp1))
    ts_labels  = [cls[0][1]] * tests_from_class
    temp2 = random.sample(set(cls).difference(temp1), tests_from_class)
    ts_samples = list(map(lambda x: extract_hog(path + x[0]), temp2))
    return (tr_samples, tr_labels, ts_samples, ts_labels)

def train(gamma_, C_, pool):
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    b_mark = time.time()    
    machine = svm.SVC(C = C_, gamma = gamma_)
    file = open(path + "gt.csv", "r")
    samples = file.read().split('\n')[1:-1]
    file.close()
    samples = list(map(lambda x: tuple(x.split(",")), samples))
    classes = []
    current = 0
    i       = 0
    while i < len(samples):  
        while i < len(samples) and samples[i][1] == samples[current][1]:
            i += 1
        classes.append(samples[current:i])
        current = i
    temp = pool.map(extract_class_samples, classes)
    tr_samples = flatten([a[0] for a in temp])
    tr_labels  = flatten([a[1] for a in temp])
    ts_samples = flatten([a[2] for a in temp])
    ts_labels  = flatten([a[3] for a in temp])
    machine.fit(tr_samples, tr_labels)
    print("\n\n" + str(time.time() - b_mark) + " elapsed.")
    print("Score: " + str(machine.score(ts_samples, ts_labels)))
    print(gamma_, C_)
    print("Save?(y/n)")
    if input() == "y":
        print("As... ", end = "")
        f = open(input() + ".svm", "wb")
        pickle.dump(machine, f)
        f.close()


pool = Pool(8)
gamma, C = map(float, input().split())
train(gamma, C, pool)
pool.close()


