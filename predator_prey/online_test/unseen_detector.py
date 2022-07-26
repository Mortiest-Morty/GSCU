import numpy as np
import torch 


class Detector:
    def __init__(self, detect_step=50):
        self.detect_step = detect_step