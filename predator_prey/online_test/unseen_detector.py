from pandas import pd
import numpy as np
import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Detector:
    def __init__(self, data_file, encoder, num_seen, detect_step=50):
        self.num_seen = num_seen
        self.estimate_num_seen = self.num_seen
        self.detect_step = detect_step
        self.offline_adv_data = self.load_offline_data(data_file)
        self.encoder = encoder
        self.init()
        self.init_thres()
    
    def init(self,):
        self.online_adv_data = []
        self.prototype_vectors = []
        self.distance = []
        for i in range(self.num_seen):
            emb = self.infer(self.offline_adv_data[i])
            proto = np.mean(emb, axis=0)
            self.prototype_vectors.append(proto)
    
    def init_thres(self,):
        # TODO: min(variance)
        self.unseen_threshold = None
    
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
    
    def detect(self,):
        current_adv_traj = self.online_adv_data[-self.detect_step:]
        current_adv_traj = np.array(current_adv_traj)
        emb = self.infer(current_adv_traj)
        proto_cur = np.mean(emb, axis=0)
        dist_list = []
        for proto_i in self.prototype_vectors:
            dist_i = np.linalg.norm(proto_cur - proto_i)
            dist_list.append(dist_i)
        dist_array = np.array(dist_list)
        self.distance.append(dist_array)
        is_unseen = np.all(dist_array > self.unseen_threshold)
        if is_unseen:
            self.estimate_num_seen += 1
        return is_unseen