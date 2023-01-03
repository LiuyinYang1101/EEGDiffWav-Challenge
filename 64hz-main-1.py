import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from util import rescale, find_max_epoch, print_size
from util import training_loss, calc_diffusion_hyperparams

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor

from models2 import EEGWav_diff as WaveNet

def train(train_loader, num_gpus, rank, group_name, output_directory, tensorboard_directory,
          ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
          learning_rate, batch_size_per_gpu):
    """
    Parameters:
    num_gpus, rank, group_name:     parameters for distributed training
    output_directory (str):         save model checkpoints to this path
    tensorboard_directory (str):    save tensorboard events to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    n_iters (int):                  number of iterations to train, default is 1M
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log, default is 100
    learning_rate (float):          learning rate
    batch_size_per_gpu (int):       batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    """

    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(wavenet_config["res_channels"],
                                           diffusion_config["T"],
                                           diffusion_config["beta_T"])
    # Create tensorboard logger.
    if rank == 0:
        tb = SummaryWriter(os.path.join('exp', local_path, tensorboard_directory))

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared output_directory ready
    output_directory = os.path.join('exp', local_path, output_directory)
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key is not "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = WaveNet(**wavenet_config).cuda()
    print_size(net)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            eeg, audio = data[0].squeeze(1).cuda(), data[1].type(torch.LongTensor).cuda()
            #print(eeg.shape,audio.shape)
            # load audio and mel spectrogram
            # mel_spectrogram = mel_spectrogram.cuda()
            # audio = audio.unsqueeze(1).cuda()

            # back-propagation
            optimizer.zero_grad()
            X = (eeg.float(), audio.float())
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams)
            # print(loss)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            # output to log
            # note, only do this on the first gpu
            if n_iter % iters_per_logging == 0 and rank == 0:
                # save training loss to tensorboard
                print("iteration: {} \treduced loss: {} \tloss: {}".format(n_iter, reduced_loss, loss.item()))
                tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1

    # Close TensorBoard.
    if rank == 0:
        tb.close()


import tensorflow as tf
import torch
import numpy as np


# Define a PyTorch dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tf_dataset, length=None):
        self.tf_dataset = tf_dataset
        if length:
            self.length = length
        else:
            self.length = 0
            for element in tf_dataset:
                self.length += 1

    def __getitem__(self, idx):
        x, y = self.tfdataset2np(self.tf_dataset.skip(idx).take(1))
        return x[0][0], y[0][0]

    def __len__(self):
        return self.length

    def tfdataset2np(self, ds):
        x, y = map(list, zip(*list(ds.as_numpy_iterator())))
        return x, y

"""Use the linear model example to train an EEGDiffWav."""
import glob
import json
import logging
import os
import tensorflow as tf
from torch.utils.data import DataLoader
from task2_regression.models.linear import simple_linear_model
from task2_regression.util.dataset_generator import RegressionDataGenerator, create_tf_dataset


def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation


if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length = 10 * 64  # 10 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64
    epochs = 100
    patience = 5
    batch_size = 1
    only_evaluate = False
    training_log_filename = "training_log.csv"
    results_filename = 'eval.json'


    # Get the path to the config file
    experiments_folder = "C:/Users/YLY/Documents/eegAudChallenge/auditory-eeg-challenge-2023-code/task2_regression"
    task_folder = os.path.dirname(experiments_folder)
    config_path = os.path.join(task_folder, 'util', 'config.json')

    # Load the config
    with open("C:/Users/YLY/Documents/eegAudChallenge/auditory-eeg-challenge-2023-code/task2_regression/util/config.json") as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test

    data_folder = os.path.join(config["dataset_folder"], config["split_folder"])
    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_linear_baseline")
    os.makedirs(results_folder, exist_ok=True)

    # create a simple linear model
    #model = simple_linear_model()
    #model.summary()
    #model_path = os.path.join(results_folder, "model.h5")

    if only_evaluate:
    #    model = tf.keras.models.load_model(model_path)
        print("evaluate")
    else:

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # Create list of numpy array files
        train_generator = RegressionDataGenerator(train_files, window_length)
        dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size)

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = RegressionDataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size)
        # Convert the TensorFlow dataset to a PyTorch dataset
        train_dataset= CustomDataset(dataset_train,340585)
        del dataset_train
        val_dataset = CustomDataset(dataset_val,38357)
        del dataset_val
        #print(train_dataset.__len__(),val_dataset.__len__())
        #print(train_dataset.__getitem__(123)[0].shape)
        train_loader = DataLoader(
                train_dataset, batch_size=16, shuffle=False
            )
        vali_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False
        )
    #### train the model
    # Parse configs. Globals nicer in this case
        with open('train.json') as f:
            data = f.read()
        config = json.loads(data)
        train_config = config["train_config"]  # training parameters
        global dist_config
        dist_config = config["dist_config"]  # to initialize distributed training
        global wavenet_config
        wavenet_config = config["wavenet_config"]  # to define wavenet
        global diffusion_config
        diffusion_config = config["diffusion_config"]  # basic hyperparameters
        global trainset_config
        trainset_config = config["trainset_config"]  # to load trainset
        global diffusion_hyperparams
        diffusion_hyperparams = calc_diffusion_hyperparams(
            **diffusion_config)  # dictionary of all diffusion hyperparameters

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        train(train_loader, 1, 0, "test", **train_config)