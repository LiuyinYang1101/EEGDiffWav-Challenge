import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from CustomDatasetPytorch import CustomProcessInOrderDataset

from util import rescale, find_max_epoch, print_size
from util import training_loss, calc_diffusion_hyperparams

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor

from WaveNet_regressor import WaveNet_regressor as WaveNet


def train(window_length, hop_length, num_gpus, rank, group_name, output_directory, tensorboard_directory,
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
    local_path = "ch{}_wavRegressor".format(wavenet_config["res_channels"])
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
    # for key in diffusion_hyperparams:
    #    if key is not "T":
    #       diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

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

    # Get the path to the config file
    experiments_folder = "C:/Users/YLY/Documents/eegAudChallenge/auditory-eeg-challenge-2023-code/task2_regression"
    task_folder = Path(experiments_folder)
    config_path = task_folder / "util/config.json"

    with open(config_path) as fp:
        config = json.load(fp)

    data_folder = Path(config["dataset_folder"]) / config["split_folder"]
    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features

    train_files = [path for path in Path(data_folder).resolve().glob("train_-_*") if
                   path.stem.split("_-_")[-1].split(".")[0] in features]

    valid_files = [path for path in Path(data_folder).resolve().glob("val_-_*") if
                   path.stem.split("_-_")[-1].split(".")[0] in features]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = CustomProcessInOrderDataset(train_files, window_length, hop_length, device)

    valid_dataset = CustomProcessInOrderDataset(valid_files, window_length, hop_length, device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False) #must be false
    vali_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False) #must be false

    # training
    n_iter = ckpt_iter + 1
    mse_loss = nn.MSELoss()
    while n_iter < n_iters + 1:
        batch_loss = 0
        no_epoch = 0
        for i, data in enumerate(train_loader, 0):
            no_epoch = i
            # get the inputs; data is a list of [inputs, labels]
            eeg, audio = data[0], data[1]
            # print(eeg.shape,audio.shape)
            # load audio and mel spectrogram
            # mel_spectrogram = mel_spectrogram.cuda()
            # audio = audio.unsqueeze(1).cuda()

            # back-propagation
            optimizer.zero_grad()
            loss = mse_loss(net(eeg), audio)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()

            # output to log
            # note, only do this on the first gpu
        if n_iter % iters_per_logging == 0 and rank == 0:
            # save training loss to tensorboard
            with torch.no_grad():
                net.eval()  # Optional when not using Model Specific layer
                valLoss = 0
                no_valid_epoch = 0
                for i, data in enumerate(vali_loader, 0):
                    # get the inputs
                    no_valid_epoch = i
                    eeg, audio = data[0], data[1]
                    # calc loss
                    loss = mse_loss(net(eeg), audio)
                    valLoss += loss.item()

            train_loss = batch_loss / no_epoch
            val_loss = valLoss / no_valid_epoch
            print("iteration: {} \ training loss: {} \ validation loss: {}".format(n_iter, train_loss, val_loss))
            print("--- %s seconds ---" % (time.time() - start_time))
            tb.add_scalar("Log-Train-Loss", np.log(train_loss), n_iter)
            tb.add_scalar("Log-Validation-Loss", np.log(val_loss), n_iter)

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


"""Example experiment for a linear baseline method."""
import glob
import json
import logging
import os
import tensorflow as tf
from torch.utils.data import DataLoader
from task2_regression.models.linear import simple_linear_model
from task2_regression.util.dataset_generator import RegressionDataGenerator, create_tf_dataset
from pathlib import Path


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
    window_length = 5 * 64  # 5s
    # Hop length between two consecutive decision windows
    hop_length = int(64 * 0.5)  # 0.5 seconds
    epochs = 100
    patience = 5
    batch_size = 1
    only_evaluate = False
    training_log_filename = "training_log.csv"
    results_filename = 'eval.json'

    # Get the path to the config file
    experiments_folder = "C:/Users/YLY/Documents/eegAudChallenge/auditory-eeg-challenge-2023-code/task2_regression"
    task_folder = Path(experiments_folder)
    config_path = task_folder / "util/config.json"

    # Load the config
    with open(config_path) as fp:
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
    # model = simple_linear_model()
    # model.summary()
    # model_path = os.path.join(results_folder, "model.h5")

    if only_evaluate:
        #    model = tf.keras.models.load_model(model_path)
        print("evaluate")
    else:

        # train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # Create list of numpy array files
        # train_generator = RegressionDataGenerator(train_files, window_length)
        # dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size)

        # Create the generator for the validation set
        # val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # val_generator = RegressionDataGenerator(val_files, window_length)
        # dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size)
        # Convert the TensorFlow dataset to a PyTorch dataset
        # train_dataset= CustomDataset(dataset_train,340585)
        # del dataset_train
        # val_dataset = CustomDataset(dataset_val,38357)
        # del dataset_val
        #### train the model
        # Parse configs. Globals nicer in this case
        with open('waveNet-regressor.json') as f:
            data = f.read()
        config = json.loads(data)
        train_config = config["train_config"]  # training parameters
        global dist_config
        dist_config = config["dist_config"]  # to initialize distributed training
        global wavenet_config
        wavenet_config = config["wavenet_config"]  # to define wavenet
        # global diffusion_config
        # diffusion_config = config["diffusion_config"]  # basic hyperparameters
        global trainset_config
        trainset_config = config["trainset_config"]  # to load trainset
        # global diffusion_hyperparams
        # diffusion_hyperparams = calc_diffusion_hyperparams(
        #    **diffusion_config)  # dictionary of all diffusion hyperparameters

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        train(window_length, hop_length, 1, 0, "test", **train_config)

    # Evaluate the model on test set
    # Create a dataset generator for each test subject
    # test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    # Get all different subjects from the test set
    # subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    # datasets_test = {}
    # Create a generator for each subject
    # for sub in subjects:
    #    files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
    #    test_generator = RegressionDataGenerator(files_test_sub, window_length)
    #    datasets_test[sub] = create_tf_dataset(test_generator, window_length, None, hop_length, 1)

    # Evaluate the model
    # evaluation = evaluate_model(model, datasets_test)

    # We can save our results in a json encoded file
    # results_path = os.path.join(results_folder, results_filename)
    # with open(results_path, "w") as fp:
    #    json.dump(evaluation, fp)
    # logging.info(f"Results saved at {results_path}")
