import sys
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid

import mixed_precision
from utils import test_model, test_decoder_model
from stats import AverageMeterSet, update_train_accuracies
from datasets import Dataset
from costs import loss_xent, loss_MSE


def _train(model, optim_inf, scheduler_inf, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device,decoder_training =  False):
    '''
    Training loop for optimizing encoder
    '''
    # If mixed precision is on, will add the necessary hooks into the model
    # and optimizer for half() conversions
    model, optim_inf = mixed_precision.initialize(model, optim_inf)
    optim_raw = mixed_precision.get_optimizer(optim_inf)

    test = test_decoder_model if model.decoder_training else test_model # Nawid - This chooses which method of testing to use

    # get target LR for LR warmup -- assume same LR for all param groups
    for pg in optim_raw.param_groups:
        lr_real = pg['lr']

    # IDK, maybe this helps?
    torch.cuda.empty_cache()

    # prepare checkpoint and stats accumulator
    next_epoch, total_updates = checkpointer.get_current_position()
    fast_stats = AverageMeterSet()
    # run main training loop
    for epoch in range(next_epoch, epochs):
        epoch_stats = AverageMeterSet()
        epoch_updates = 0
        time_start = time.time()

        for _, ((images1, images2), labels) in enumerate(train_loader): # Nawid - obtains the images and the labels
            # get data and info about this minibatch
            labels = torch.cat([labels, labels]).to(device)
            images1 = images1.to(device)
            images2 = images2.to(device)
            # run forward pass through model to get global and local features
            res_dict = model(x1=images1, x2=images2, class_only=False)

            # compute costs for all self-supervised tasks
            loss_g2l = (res_dict['g2l_1t5'] +
                        res_dict['g2l_1t7'] +
                        res_dict['g2l_5t5']) # Nawid - loss for the global to local features predictions
            loss_inf = loss_g2l + res_dict['lgt_reg']

            if model.decoder_training:
                image_reconstructions =  res_dict['decoder_output']
                target_images = torch.cat([images1,images2]) # Nawid - Concatenate both batches along the dimension of number of training examples
                auxiliary_loss =  loss_MSE(image_reconstructions, target_images)
                epoch_stats.update_dict({'loss_decoder': auxiliary_loss.item()},n =1)

            else:
                # compute loss for online evaluation classifiers
                lgt_glb_mlp, lgt_glb_lin = res_dict['class']
                auxiliary_loss = (loss_xent(lgt_glb_mlp, labels) + # Nawid - Loss for the classifier terms
                            loss_xent(lgt_glb_lin, labels))
                epoch_stats.update_dict({
                    'loss_cls': auxiliary_loss.item()
                }, n=1)
                update_train_accuracies(epoch_stats, labels, lgt_glb_mlp, lgt_glb_lin)


            # do hacky learning rate warmup -- we stop when LR hits lr_real
            if (total_updates < 500):
                lr_scale = min(1., float(total_updates + 1) / 500.)
                for pg in optim_raw.param_groups:
                    pg['lr'] = lr_scale * lr_real

            # reset gradient accumlators and do backprop
            loss_opt = loss_inf + auxiliary_loss # Nawid - Total loss is the loss from the global to local prediction as well as the loss from the classifier predictions
            optim_inf.zero_grad()
            mixed_precision.backward(loss_opt, optim_inf)  # backwards with fp32/fp16 awareness
            optim_inf.step()

            # record loss and accuracy on minibatch
            epoch_stats.update_dict({ # Nawid - Changed the update so that the auxillary loss is calculated above
                'loss_inf': loss_inf.item(),
                'loss_g2l': loss_g2l.item(),
                'lgt_reg': res_dict['lgt_reg'].item(),
                'loss_g2l_1t5': res_dict['g2l_1t5'].item(),
                'loss_g2l_1t7': res_dict['g2l_1t7'].item(),
                'loss_g2l_5t5': res_dict['g2l_5t5'].item()
            }, n=1)


            # shortcut diagnostics to deal with long epochs
            total_updates += 1
            epoch_updates += 1
            if (total_updates % 100) == 0:
                # IDK, maybe this helps?
                torch.cuda.empty_cache()
                time_stop = time.time()
                spu = (time_stop - time_start) / 100.
                print('Epoch {0:d}, {1:d} updates -- {2:.4f} sec/update'
                      .format(epoch, epoch_updates, spu))
                time_start = time.time()
            if (total_updates % 500) == 0:
                # record diagnostics
                eval_start = time.time()
                fast_stats = AverageMeterSet() # Nawid - This is short term stats which are reset regularly
                test(model, test_loader, device, fast_stats, max_evals=100000) # Nawd - test is chosen to be test_decoder_model or test_model at the start of the function based on whether decoder training is occuring or not

                stat_tracker.record_stats(
                    fast_stats.averages(total_updates, prefix='fast/')) # Nawid - This is used to record the data in tensorboard, where the average of the different values are placed in tensorboard,total_updates is the index which is used to place information in tensorbard i believe
                eval_time = time.time() - eval_start
                stat_str = fast_stats.pretty_string(ignore=model.tasks)
                stat_str = '-- {0:d} updates, eval_time {1:.2f}: {2:s}'.format(
                    total_updates, eval_time, stat_str)
                print(stat_str)

        # update learning rate
        scheduler_inf.step(epoch)
        test(model, test_loader, device, epoch_stats, max_evals=500000)
        epoch_str = epoch_stats.pretty_string(ignore=model.tasks)
        diag_str = '{0:d}: {1:s}'.format(epoch, epoch_str)
        print(diag_str)
        sys.stdout.flush()
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix='costs/')) # Nawid - This is used to update long-term stats which are used for a long-period of time
        checkpointer.update(epoch + 1, total_updates)


def train_self_supervised(model, learning_rate, dataset, train_loader,
                          test_loader, stat_tracker, checkpointer, log_dir, device):
    # configure optimizer
    mods_inf = [m for m in model.info_modules] # Nawid - inf stands for info_NCE
    if model.decoder_training: # Nawid - Additional parameter to control which auxillary loss used
        mods_aux = [m for m in model.decoder_modules]
    else:
        mods_aux = [m for m in model.class_modules]

    mods_to_opt = mods_inf + mods_aux
    optimizer = optim.Adam(
        [{'params': mod.parameters(), 'lr': learning_rate} for mod in mods_to_opt],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    # configure learning rate schedulers for the optimizers
    if dataset in [Dataset.C10, Dataset.C100, Dataset.STL10]:
        scheduler = MultiStepLR(optimizer, milestones=[250, 280], gamma=0.2)
        epochs = 300
    else:
        # best imagenet results use longer schedules...
        # -- e.g., milestones=[60, 90], epochs=100
        scheduler = MultiStepLR(optimizer, milestones=[30, 45], gamma=0.2)
        epochs = 50
    # train the model
    _train(model, optimizer, scheduler, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device)
