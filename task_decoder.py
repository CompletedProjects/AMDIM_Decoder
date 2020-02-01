import sys
import time

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid, save_image

import mixed_precision
from utils import weight_init, test_model, flatten, test_decoder_model, save_reconstructions
from stats import AverageMeterSet, update_train_accuracies
from datasets import Dataset
from costs import loss_xent,loss_MSE # Nawid - Added loss_MSE to calculate the loss

def _train(model, optimizer, scheduler, checkpointer, epochs, train_loader,
           test_loader, stat_tracker, log_dir, device):
    '''
    Training loop to train classifiers on top of an encoder with fixed weights.
    -- e.g., use this for eval or running on new data
    '''
    # If mixed precision is on, will add the necessary hooks into the model and
    # optimizer for half precision conversions
    model, optimizer = mixed_precision.initialize(model, optimizer)
    # ...
    time_start = time.time()
    total_updates = 0
    next_epoch, total_updates = checkpointer.get_current_position(decoder=True) # Nawid - I think if I am continuing training, then it finds the current epoch it is on
    for epoch in range(next_epoch, epochs):
        epoch_updates = 0
        epoch_stats = AverageMeterSet()
        #for _, ((images1, images2), labels) in enumerate(test_loader): # Nawid - loads the two different images
        for _, ((images1,images2),labels) in enumerate(train_loader): # Nawid - loads the two different images
            # get data and info about this minibatch
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            # run forward pass through model and collect activations
            res_dict = model(x1=images1, x2=images2, decoder_only=True) # Nawid - Only requires the first input, produces the dictionary of outputs which should be the encoder value and a decoded output

            image_reconstructions = res_dict['decoder_output'] # Nawid-  Obtains the logits from the mlp and the linear

            # compute total loss for optimization
            loss = loss_MSE(images1, image_reconstructions) # Nawid-  Compute the loss using the mlp and the linear layer - there is no loss term related to the encoder and the input to the loss term is the logits which

            # do optimizer step for encoder
            optimizer.zero_grad()
            mixed_precision.backward(loss, optimizer)  # special mixed precision stuff
            optimizer.step() # Nawid - Gradient step
            # record loss and accuracy on minibatch
            epoch_stats.update('loss', loss.item(), n=1)
            # - NEED TO  CHANGE THIS FOR THE DECODER update_train_accuracies(epoch_stats, labels, lgt_glb_mlp, lgt_glb_lin) # Nawid -  updates the accuracies

            # shortcut diagnostics to deal with long epochs
            total_updates += 1
            epoch_updates += 1
            if (total_updates % 100) == 0:
                save_reconstructions(images1,image_reconstructions)

                time_stop = time.time()
                spu = (time_stop - time_start) / 100.

                print('Epoch {0:d}, {1:d} updates -- {2:.4f} sec/update'
                      .format(epoch, epoch_updates, spu))
                time_start = time.time()

        # step learning rate scheduler
        scheduler.step(epoch)
        # record diagnostics
        test_decoder_model(model, test_loader, device, epoch_stats, max_evals=500000) # Nawid - NEED TO CHANGE FOR DECODER I BELIEVE
        epoch_str = epoch_stats.pretty_string(ignore=model.tasks)
        diag_str = '{0:d}: {1:s}'.format(epoch, epoch_str)
        print(diag_str)
        sys.stdout.flush()
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix='decoder/')) # NAWID - NEED TO CHANGE FOR DECODER
        checkpointer.update(epoch + 1, total_updates, decoder=True) # Nawid - Updates the decoder


def train_decoder(model, learning_rate, dataset, train_loader,
                      test_loader, stat_tracker, checkpointer, log_dir, device): # Nawid - Sets up the classifier parameter and then trains it at the the end
    # retrain the evaluation classifiers using the trained feature encoder
    for mod in model.decoder_modules: # Nawid - This only looks at the decoder modules
        # reset params in the evaluation classifiers
        mod.apply(weight_init) # Nawid -Reset parametes
    mods_to_opt = [m for m in model.decoder_modules] # Nawid - Makes a list of the modules to perform the optimisation on which in this case is just the classifier modules
    # configure optimizer
    optimizer = optim.Adam(
        [{'params': mod.parameters(), 'lr': learning_rate} for mod in mods_to_opt], # Nawid- applies the optimisation just to the classsifier modules which is specificed in mods_to_opt
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    # configure learning rate schedulers
    if dataset in [Dataset.C10, Dataset.C100, Dataset.STL10]: # Nawid - Sets a scheduler based on the specific dataset
        scheduler = MultiStepLR(optimizer, milestones=[80, 110], gamma=0.2)
        epochs = 120
    elif dataset == Dataset.IN128:
        scheduler = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.2)
        epochs = 30
    elif dataset == Dataset.PLACES205:
        scheduler = MultiStepLR(optimizer, milestones=[7, 12], gamma=0.2)
        epochs = 15
    # retrain the model
    _train(model, optimizer, scheduler, checkpointer, epochs, train_loader,
           test_loader, stat_tracker, log_dir, device)
