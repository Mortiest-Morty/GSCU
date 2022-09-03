import numpy as np
import pandas as pd
import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Detector:
    def __init__(self, data_file, encoder, num_seen, detect_step=50):
        self.num_detect = 0
        self.num_seen = num_seen
        self.estimate_num_seen = self.num_seen
        self.detect_step = detect_step
        self.offline_adv_data = self.load_offline_data(data_file)
        self.offline_adv_emb = []  # self.num_seen * [bs, 2]
        self.encoder = encoder
        self.online_adv_data = []
        self.prototypes = []
        self.radius = []
        self.seen_threshold = 0.5
        self.online_detect_result = []
        self.online_detect_is_unseen = []
        self.online_adv_label = []
        self.init()
    
    def init(self,):
        for i in range(self.num_seen):
            # use all the train data to calculate prototypes
            emb = self.infer(self.offline_adv_data[i])
            self.offline_adv_emb.append(emb)
            proto = np.mean(emb, axis=0, keepdims=True)  # [1, 2]
            self.prototypes.append(proto)
            rad = np.std(emb)
            self.radius.append(rad)
            print("i:", i, ",proto:", proto, ",rad:", rad)
    
    def infer(self, tau_vec):
        tau_vec_tensor = torch.tensor(tau_vec, dtype=torch.float).to(device)
        emb_tensor = self.encoder(tau_vec_tensor)  # [bs, 2]
        emb = emb_tensor.cpu().detach().numpy()
        return emb
    
    def load_offline_data(self, data_file):
        data = pd.read_pickle(data_file)
        tau_dict = data['data_tau']
        for key in tau_dict:
            tau_dict[key] = np.array(tau_dict[key])
        return tau_dict
    
    def add_adv_traj(self, adv_traj):
        self.online_adv_data.append(adv_traj)
    
    def add_adv_label(self, adv_label):
        self.online_adv_label.append(adv_label)

    def detect(self,):
        current_adv_traj = self.online_adv_data[-self.detect_step:]
        current_adv_traj = np.array(current_adv_traj)
        cur_emb = self.infer(current_adv_traj)  # [detect_step, 2]
        is_seen_list = []
        for i, proto in enumerate(self.prototypes):
            dist = np.linalg.norm(cur_emb - proto, axis=1)  # [detect_step, ]
            in_count = np.sum(dist <= self.radius[i])
            is_seen = (in_count / len(dist)) >= self.seen_threshold
            is_seen_list.append(is_seen)
        if True in is_seen_list:
            is_unseen = False
        else:
            is_unseen = True
        detect_result = np.array(is_seen_list, dtype=int)
        self.online_detect_is_unseen.append(is_unseen)
        self.online_detect_result.append(detect_result)
        if is_unseen:
            self.estimate_num_seen += 1
        self.num_detect += 1
        print("label:", self.online_adv_label[-1])
        print("detect:", self.online_detect_result[-1])
        return is_unseen
    
    def cal_accuracy(self,):
        online_adv_label = np.array(self.online_adv_label)
        label_is_unseen = online_adv_label >= self.num_seen
        detect_is_unseen = np.array(self.online_detect_is_unseen)
        accuracy = np.sum(label_is_unseen == detect_is_unseen) / len(label_is_unseen)
        return accuracy, self.num_detect