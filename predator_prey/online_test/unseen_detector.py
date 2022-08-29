import numpy as np
import torch 


class Detector:
    def __init__(self, detect_step=50):
        self.detect_step = detect_step
        self.offline_adv_data = self.load_offline_data()
        self.online_adv_data = []
        self.prototype_vectors = []
    
    def load_offline_data(self,):
        pass
    
    def add_adv_traj(self,):
        pass
    
    def detect(self,):
        pass