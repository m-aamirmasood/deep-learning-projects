import os
import time
import pickle
import numpy as np
import cv2
from viola_jones import ViolaJones

def v_training(a):
    classifier = ViolaJones(T=a)
    with open('Train.pkl', 'rb') as f:
        training = pickle.load(f)
    classifier.training(training, 850, 850)
    classifier.save(str(a))

if __name__ == '__main__':
    v_training(45)
