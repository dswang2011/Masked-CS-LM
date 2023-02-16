
import torch

import os
import argparse
from torch.utils.data import DataLoader

import torch
import pickle
from utils.params import Params
# from torch_geometric.transforms import NormalizeFeatures
import dataSetup
from LMs import trainer
import LMs

def parse_args(config_path):
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()

if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/finetune.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', params.device)

    # section 2, objective function and output dim/ move to trainer
    # this is usually covered by huggingface models
    mydata = dataSetup.setup(params)
    print(mydata.trainable_dataset)
    # section 3, model, loss function, and optimizer
    if bool(params.continue_train):
        print('continue based on:', params.continue_with_model)
        model_params = pickle.load(open(os.path.join(params.continue_with_model,'config.pkl'),'rb'))
        model = LMs.setup(model_params).to(params.device)
        model.load_state_dict(torch.load(os.path.join(params.continue_with_model,'model')))
    else:
        model = LMs.setup(params).to(params.device)


    # section 4, saving path for output model
    params.dir_path = trainer.create_save_dir(params)    # prepare dir for saving best models

    # section 5, load data; prepare output_dim/num_labels, id2label, label2id for section3;
    print(mydata.trainable_dataset)
    best_f1 = trainer.finetune(params, model, mydata)

    # section 5, inference only (on test_dataset)
    # inferencer.inference(params,model,mydata,'v3_base_benchmark_Jan_1pm.json')


