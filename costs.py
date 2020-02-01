import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tanh_clip(x, clip_val=10.): # Nawid- Clips the output of a tanh
    '''
    soft clip values to the range [-clip_val, +clip_val]
    '''
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


def loss_xent(logits, labels, ignore_index=-1):
    '''
    compute multinomial cross-entropy, for e.g. training a classifier.
    '''
    xent = F.cross_entropy(tanh_clip(logits, 10.), labels,
                           ignore_index=ignore_index) # Nawid - Calculates the cross entropy between a clipped logits and the labels
    lgt_reg = 1e-3 * (logits**2.).mean() # Nawid - regularisation term which penailise the square matching scores
    return xent + lgt_reg

def loss_MSE(input_image, decoded_output): # Nawid - Loss function for the mse loss
    '''
    Computes the MSE, for e.g training a decoderself.
    '''
    loss = F.mse_loss(input_image,decoded_output)
    return loss

def loss_BCE(input_image, decoded_output): # Nawid - Loss function for the mse loss
    '''
    Computes the Binary cross entropy , for e.g training a decoderself.
    '''
    loss = F.binary_cross_entropy(input_image,decoded_output)
    return loss


class NCE_MI_MULTI(nn.Module):
    def __init__(self, tclip=20.):
        super(NCE_MI_MULTI, self).__init__()
        self.tclip = tclip

    def _model_scores(self, r_src, r_trg, mask_mat):
        '''
        Compute the NCE scores for predicting r_src->r_trg.

        Input:
          r_src    : (n_batch_gpu, n_rkhs)
          r_trg    : (n_rkhs, n_batch * n_locs)
          mask_mat : (n_batch_gpu, n_batch)
        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
        '''
        n_batch_gpu = mask_mat.size(0)
        n_batch = mask_mat.size(1)
        n_locs = r_trg.size(1) // n_batch # Nawid -  I believe this is the number of spatial locations for the local features
        n_rkhs = r_src.size(1) # Nawid - I believe this controls the output dimension of the embedding
        # reshape mask_mat for ease-of-use
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, n_locs).float() # Nawid - This makes it so that the mask tensor is (n_batch_gpu, n_batch,1), then changes it to (n_batch_gpu, n_batch,n_loc) so it basically copied the tesnor of (n_batch_gpu, n_batch) across the 3rd dimension - copied the 2D tensor across the depth
        mask_neg = 1. - mask_pos # Nawid - The positive mask is 1 in some locations and zero in others. Negative mask is made by 1 - positve mask so it will be 1 in locations where the positve mask is zero

        # compute src->trg raw scores for batch on this gpu
        raw_scores = torch.mm(r_src, r_trg).float() # Nawid - Matrix multiplication between source and target values - seems to correspond to equation 2
        raw_scores = raw_scores.reshape(n_batch_gpu, n_batch, n_locs)
        raw_scores = raw_scores / n_rkhs**0.5
        lgt_reg = 5e-2 * (raw_scores**2.).mean() # Nawid - regularisaion term which penalises squared matching scores
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip) # Nawid - Soft clipping non-linearity

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # (n_batch_gpu, n_locs)
        pos_scores = (mask_pos * raw_scores).sum(dim=1) # Nawid - Scores for all positve examples as he mask_pos will be 0 at any position where a positive example was not chosen - There should only be 1 postive value since the mask is initially chosen to be zeroes everywhere except 1 position
        # (n_batch_gpu, n_batch, n_locs)
        neg_scores = (mask_neg * raw_scores) - (self.tclip * mask_pos) # Nawid - Obtains all the scores for the negative values and sets the value for the positive value to the minimum score
        # (n_batch_gpu, n_batch * n_locs)
        neg_scores = neg_scores.reshape(n_batch_gpu, -1)
        # (n_batch_gpu, n_batch * n_locs)
        mask_neg = mask_neg.reshape(n_batch_gpu, -1)
        '''
        for each set of positive examples P_i, compute the max over scores
        for the set of negative samples N_i that are shared across P_i
        '''
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0] # Nawid - Maximum value for the negative (first line of pseudocode)
        '''
        compute a "partial, safe sum exp" over each negative sample set N_i,
        to broadcast across the positive samples in P_i which share N_i
        -- size will be (n_batch_gpu, 1)
        '''
        neg_sumexp = \
            (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True) # Nawid - I believe the minus neg_maxes is used for numeric stability
        '''
        use broadcasting of neg_sumexp across the scores in P_i, to compute
        the log-sum-exps for the denominators in the NCE log-softmaxes
        -- size will be (n_batch_gpu, n_locs)
        '''
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp) # Nawid - This corresponds to the denominator term of equation 2
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes # Nawid -  I believe this corresponds to the numerator term
        # compute the final log-softmax scores for NCE...
        nce_scores = pos_shiftexp - all_logsumexp
        return nce_scores, pos_scores, lgt_reg

    def _loss_g2l(self, r_src, r_trg, mask_mat):
        # compute the nce scores for these features
        nce_scores, raw_scores, lgt_reg = \
            self._model_scores(r_src, r_trg, mask_mat)
        loss_g2l = -nce_scores.mean() # Nawid - Computes the loss (equation 2)
        return loss_g2l, lgt_reg

    def forward(self, r1_src_1, r5_src_1, r1_src_2, r5_src_2,
                r5_trg_1, r7_trg_1, r5_trg_2, r7_trg_2, mask_mat, mode):
        assert(mode in ['train', 'viz'])
        if mode == 'train':
            # compute values required for visualization
            if mask_mat.sum().item() < 1e-1:
                # hack for avoiding nce computation on cuda:0
                loss_1t5 = mask_mat.detach().mean()
                loss_1t7 = mask_mat.detach().mean()
                loss_5t5 = mask_mat.detach().mean()
                lgt_reg = mask_mat.detach().mean()
            else:
                # compute costs for 1->5 prediction
                loss_1t5_1, lgt_1t5_1 = self._loss_g2l(r1_src_1, r5_trg_2[0], mask_mat) # Nawid - Using global features r1 of image 1 to make predictions of local 5 of image 2
                loss_1t5_2, lgt_1t5_2 = self._loss_g2l(r1_src_2, r5_trg_1[0], mask_mat) # Nawid - Using global features r1 of image 2 to make predictions of local 5 of image 1
                # compute costs for 1->7 prediction
                loss_1t7_1, lgt_1t7_1 = self._loss_g2l(r1_src_1, r7_trg_2[0], mask_mat) # Nawid - Using global features r1 of image 1 to make predictions of local 7 of image 2
                loss_1t7_2, lgt_1t7_2 = self._loss_g2l(r1_src_2, r7_trg_1[0], mask_mat) # Nawid - Using global features r1 of image 2 to make predictions of local 7 of image 1
                # compute costs for 5->5 prediction
                loss_5t5_1, lgt_5t5_1 = self._loss_g2l(r5_src_1, r5_trg_2[0], mask_mat)
                loss_5t5_2, lgt_5t5_2 = self._loss_g2l(r5_src_2, r5_trg_1[0], mask_mat)
                # combine costs for optimization
                loss_1t5 = 0.5 * (loss_1t5_1 + loss_1t5_2)
                loss_1t7 = 0.5 * (loss_1t7_1 + loss_1t7_2)
                loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)
                lgt_reg = 0.5 * (lgt_1t5_1 + lgt_1t5_2 + lgt_1t7_1 +
                                 lgt_1t7_2 + lgt_5t5_1 + lgt_5t5_2) # Nawid - Calculates regularisaion terms
            return loss_1t5, loss_1t7, loss_5t5, lgt_reg
        else:
            # compute values to use for visualizations
            nce_scores, raw_scores, lgt_reg = \
                self._model_scores(r1_src_1, r7_trg_2[0], mask_mat)
            return nce_scores, raw_scores


class LossMultiNCE(nn.Module):
    '''
    Input is fixed as r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2.
    '''

    def __init__(self, tclip=10.):
        super(LossMultiNCE, self).__init__()
        # initialize the dataparallel nce computer (magic!)
        self.nce_func = NCE_MI_MULTI(tclip=tclip)
        self.nce_func = nn.DataParallel(self.nce_func)
        # construct masks for sampling source features from 5x5 layer
        masks_r5 = np.zeros((5, 5, 1, 5, 5)) # Nawid - 5 dimensional tensor of zeros ( with the third dimension having a dimension of 1 so it is essentially 4 dimension)
        for i in range(5):
            for j in range(5):
                masks_r5[i, j, 0, i, j] = 1 # Nawid - Sets the values in the mask equal to 1
        masks_r5 = torch.tensor(masks_r5).type(torch.uint8)
        masks_r5 = masks_r5.reshape(-1, 1, 5, 5) # Nawid - Makes it into a 4D tensor with shape (25,1,5,5)
        self.masks_r5 = nn.Parameter(masks_r5, requires_grad=False)

    def _sample_src_ftr(self, r_cnv, masks): # Nawid - sampling source features from a 5 x5 feature map
        # get feature dimensions
        n_batch = r_cnv.size(0)
        n_rkhs = r_cnv.size(1)
        if masks is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, masks.size(0), (n_batch,)) # Nawid - samples a integer from from 0 to masks.size(0) ( which should be a value of 25 I believe) and the number of samples is n_batch
            r_cnv = torch.masked_select(r_cnv, masks[mask_idx]) # Nawid -Obtain values which are indicated from the mask indices
        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(n_batch, n_rkhs) # Nawod - Shape into number of batches and n_rkhs which is the size of the feature vector - so this gets several different feature vectors
        return r_vec

    def forward(self, r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2): # Nawid - i beleive rk denotes which layer the feature of the image is taken from
        '''
        Compute nce infomax costs for various combos of source/target layers.

        Compute costs in both directions, i.e. from/to both images (x1, x2).

        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''
        # compute feature dimensions
        n_batch = int(r1_x1.size(0))
        n_rkhs = int(r1_x1.size(1)) # Nawid - Feature vector size
        # make masking matrix to help compute nce costs
        mask_mat = torch.eye(n_batch).cuda()

        # sample "source" features for glb->lcl predictions
        r1_src_1 = self._sample_src_ftr(r1_x1, None) # Nawid - Sample source from the gloabl feature vector ( no mask needed i believe since there is only 1 global feature vector per image)
        r5_src_1 = self._sample_src_ftr(r5_x1, self.masks_r5) # Nawid - Sample local features from image 1 using mask
        r1_src_2 = self._sample_src_ftr(r1_x2, None)
        r5_src_2 = self._sample_src_ftr(r5_x2, self.masks_r5)  # Nawid - Sample local features from image 2 using mask

        # before shape: (n_batch, n_rkhs, n_dim, n_dim)
        r5_trg_1 = r5_x1.permute(1, 0, 2, 3).reshape(n_rkhs, -1)
        r7_trg_1 = r7_x1.permute(1, 0, 2, 3).reshape(n_rkhs, -1)
        r5_trg_2 = r5_x2.permute(1, 0, 2, 3).reshape(n_rkhs, -1)
        r7_trg_2 = r7_x2.permute(1, 0, 2, 3).reshape(n_rkhs, -1)
        # after shape: (n_rkhs, n_batch * n_dim * n_dim)

        # compute global->local scores and nce costs via nn.Dataparallel
        n_gpus = torch.cuda.device_count()
        r5_trg_1 = r5_trg_1.unsqueeze(dim=0).expand(n_gpus, -1, -1)
        r7_trg_1 = r7_trg_1.unsqueeze(dim=0).expand(n_gpus, -1, -1)
        r5_trg_2 = r5_trg_2.unsqueeze(dim=0).expand(n_gpus, -1, -1)
        r7_trg_2 = r7_trg_2.unsqueeze(dim=0).expand(n_gpus, -1, -1)

        # we're going to hackishly cut mem use on device cuda:0
        if n_gpus >= 4:
            assert (n_batch % (n_gpus - 1) == 0), 'n_batch: {}, n_gpus: {}'.format(n_batch, n_gpus)
            # expand tensors with dummy chunks so cuda:0 can skip compute
            chunk_size = n_batch // (n_gpus - 1)
            dummy_chunk = torch.zeros_like(r1_src_1[:chunk_size])
            r1_src_1 = torch.cat([dummy_chunk, r1_src_1], dim=0)
            r5_src_1 = torch.cat([dummy_chunk, r5_src_1], dim=0)
            r1_src_2 = torch.cat([dummy_chunk, r1_src_2], dim=0)
            r5_src_2 = torch.cat([dummy_chunk, r5_src_2], dim=0)
            # ...
            dummy_chunk = torch.zeros_like(mask_mat[:chunk_size])
            mask_mat = torch.cat([dummy_chunk, mask_mat], dim=0)

        # compute nce for multiple infomax costs across multiple GPUs
        # -- we do some hacky stuff to minimize compute/mem costs for cuda:0
        loss_1t5, loss_1t7, loss_5t5, lgt_reg = \
            self.nce_func(r1_src_1, r5_src_1, r1_src_2, r5_src_2,
                          r5_trg_1, r7_trg_1, r5_trg_2, r7_trg_2,
                          mask_mat, mode='train')

        # adjust cost weight to compensate for hacky skip of cuda:0
        if n_gpus >= 4:
            rescale = float(n_gpus) / float(n_gpus - 1)
            loss_1t5 = rescale * loss_1t5.mean()
            loss_1t7 = rescale * loss_1t7.mean()
            loss_5t5 = rescale * loss_5t5.mean()
            lgt_reg = rescale * lgt_reg.mean()
        else:
            loss_1t5 = loss_1t5.mean()
            loss_1t7 = loss_1t7.mean()
            loss_5t5 = loss_5t5.mean()
            lgt_reg = lgt_reg.mean()
        return loss_1t5, loss_1t7, loss_5t5, lgt_reg
