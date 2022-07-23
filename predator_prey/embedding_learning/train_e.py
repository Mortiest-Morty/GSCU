import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import torch
import pickle
import logging
import argparse
import pandas as pd 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss,NLLLoss,CrossEntropyLoss,MSELoss
from embedding_learning.dataset_e import OpponentEDataset
from embedding_learning.opponent_models_e import Encoder
from utils.config_predator_prey import Config

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

adv_pool = Config.ADV_POOL_SEEN

def main(version):

    data_dir = Config.DATA_DIR
    model_dir = Config.VAE_MODEL_DIR

    train_data_file = data_dir + 'e_data_simple_tag_' + version + '.p'
    test_data_file = data_dir + 'e_data_simple_tag_' + version + '_test.p'
    batch_size = 1024
    gpuid = [0] 
    epochs = 30
    learning_rate = 1e-4
    obs_dim = 16
    action_dim = 7
    input_dim = int((obs_dim+action_dim)*50)
    hidden_dim = Config.HIDDEN_DIM
    latent_dim = Config.LATENT_DIM

    train_dset = OpponentEDataset(train_data_file, train=True)
    train_data_loader = DataLoader(train_dset,
                            batch_size = batch_size,
                            shuffle = True,
                            )
    test_dset = OpponentEDataset(test_data_file, train=False)
    test_data_loader = DataLoader(test_dset,
                            batch_size = batch_size,
                            shuffle = False,
                            )
    logging.info("finish loading data!")
    logging.info("total num train iter: {0}.".format(len(train_data_loader)))
    logging.info("total num test iter: {0}.".format(len(test_data_loader)))
    use_cuda = (len(gpuid) >= 1)

    encoder = Encoder(input_dim, hidden_dim, latent_dim)

    if use_cuda > 0:
        encoder.cuda()
    
    loss_func = MSELoss()
    parameters = list(encoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    for epoch_i in range(0, epochs):
        logging.info("At {0}-th epoch.".format(epoch_i))

        # training
        encoder.train()
        train_loss = 0.0
        for it, data in enumerate(train_data_loader, 0):
            data_i, data_j, data_plus, data_r, data_minus = data

            if use_cuda:
                data_i, data_j, data_plus, data_r, data_minus = Variable(data_i).cuda(), Variable(data_j).cuda(), Variable(data_plus).cuda(), Variable(data_r).cuda(), Variable(data_minus).cuda()
            else:
                data_i, data_j, data_plus, data_r, data_minus = Variable(data_i), Variable(data_j), Variable(data_plus), Variable(data_r), Variable(data_minus)

            emb_plus = encoder(data_plus)
            emb_r = encoder(data_r)
            emb_minus = encoder(data_minus)

            loss = torch.pow(1 + torch.exp(loss_func(emb_r, emb_minus) - loss_func(emb_r, emb_plus)), -2)

            train_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_avg_loss = train_loss / (len(train_dset) / batch_size)
        train_avg_loss = train_avg_loss.cpu().detach().numpy()
        logging.info("Average training loss value per instance is {0} at the end of epoch {1}".format(train_avg_loss, epoch_i))

        ################################################
        # testing
        encoder.eval()
        test_loss = 0.0

        for it, data in enumerate(test_data_loader, 0):
            data_i, data_j, data_plus, data_r, data_minus = data

            if use_cuda:
                data_i, data_j, data_plus, data_r, data_minus = Variable(data_i).cuda(), Variable(data_j).cuda(), Variable(data_plus).cuda(), Variable(data_r).cuda(), Variable(data_minus).cuda()
            else:
                data_i, data_j, data_plus, data_r, data_minus = Variable(data_i), Variable(data_j), Variable(data_plus), Variable(data_r), Variable(data_minus)
            
            emb_plus = encoder(data_plus)
            emb_r = encoder(data_r)
            emb_minus = encoder(data_minus)

            loss = torch.pow(1 + torch.exp(loss_func(emb_r, emb_minus) - loss_func(emb_r, emb_plus)), -2)
            
            test_loss += loss.data

        test_avg_loss = test_loss / (len(test_dset) / batch_size)
        test_avg_loss = test_avg_loss.cpu().detach().numpy()
        logging.info("Average testing loss value per instance is {0} at the end of epoch {1}".format(test_avg_loss, epoch_i))
    
    torch.save(encoder.state_dict(), model_dir+'encoder_e_param_'+version+'_'+str(epoch_i)+'.pt')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--version', default='v0')
    args = parser.parse_args()
    main(args.version)
