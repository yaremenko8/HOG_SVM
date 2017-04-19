from sklearn import svm
from hog import extract_hog
import pickle


def fit_and_classify(machine_file, samples):
    f = open(machine_file, "rb")
    machine = pickle.load(f)
    f.close()
    descriptors = []
    for i in samples:
        descriptors.append(extract_hog(i))
    return machine.predict(descriptors)

