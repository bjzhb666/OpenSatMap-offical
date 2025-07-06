# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class AELoss(nn.Module):
    """Associate embedding loss.

    Args:
        loss_weight (float): Weight of AE loss.
        pull_push_weight (float): Weight of pull and push loss.
    """

    def __init__(self, loss_weight=1.0, push_loss_factor=0.1, minimum_instance_pixels=0,
                 loss_name='loss_ae'):
        super(AELoss, self).__init__()
        self.loss_weight = loss_weight
        self.push_loss_factor = push_loss_factor
        self.minimum_instance_pixels = minimum_instance_pixels
        self._loss_name = loss_name
    
    def _calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1,keepdim=True)
        rec = rec/rec.norm(dim=-1,keepdim=True)
        rec_loss = (1-(target *rec).sum(-1)).mean()
        
        return rec_loss
    
    def _ae_loss_per_image(self, pred, target, embedding_dim, ignore_position):
        '''Calculate the AE loss for each image in the batch
        Args:
            pred (Tensor): The predicted embedding map. Shape (L, H, W).
            target (Tensor): The target embedding map. Shape (1, H, W).
            embedding_dim (int): The embedding dimension.
            ignore_position (Tensor): The mask of ignore position. Shape (1, H, W).
        '''
        pred = pred.view(embedding_dim, -1) # (L, H*W)
        target = target.view(1, -1).squeeze() # (H*W)
        ignore_mask = ignore_position.view(-1) ==0 # Flatten the ignore position mask: 1 for valid, 0 for ignore

        # use ignore mask to filter out the ignore position, and also ignore the background
        valid_mask = ignore_mask & (target != 0)
        unique_instances = target[valid_mask].unique() # get the unique instance ids

        # get the number of valid instances
        N = unique_instances.size(0)

        instance_kpt_embeddings = []
        instance_tags = []

        for instance_id in unique_instances:
            mask = (target == instance_id) & valid_mask
            if mask.sum() > self.minimum_instance_pixels:  # 确保至少有一个有效像素
                kpt_embedding = pred[:, mask]
                instance_kpt_embeddings.append(kpt_embedding)
                instance_tags.append(kpt_embedding.mean(dim=1))

        N = len(instance_kpt_embeddings) # 重新计算有效实例数
        # print(N)
        if N == 0:
            # warnings.warn('No valid instance in the image')
            useless_gradients = pred.mean() 
            pull_loss = torch.tensor(0.0, requires_grad=True).to(pred.device) * useless_gradients
            push_loss = torch.tensor(0.0, requires_grad=True).to(pred.device) * useless_gradients
        else:
            # import pdb; pdb.set_trace()
            # pull_loss = sum(
            #     F.mse_loss(kpt_embedding, tag.view(-1, 1).expand_as(kpt_embedding))
            # for kpt_embedding, tag in zip(instance_kpt_embeddings, instance_tags)
            # )
            pull_loss = 0.0
            for kpt_embedding, tag in zip(instance_kpt_embeddings, instance_tags):
                kpt_embedding = kpt_embedding.permute(1, 0)  # (N, L)
                tag = tag.unsqueeze(0)  # (1, L)
                pull_loss += self._calculate_rec_loss(kpt_embedding, tag.repeat(kpt_embedding.size(0), 1))
            
            if N == 1:
                push_loss = torch.tensor(0.0, requires_grad=True).to(pred.device)  # 如果只有一个有效实例或没有有效实例
            else:
                tag_mat = torch.stack(instance_tags)  # (N, L)
                tag_mat = tag_mat / tag_mat.norm(dim=-1, keepdim=True)  # normalize the tag
                # diff = (tag_mat[None] - tag_mat[:, None])**2  # (N, N, L)
                diff = 1 - (tag_mat[None] * tag_mat[:, None]).sum(dim=-1)  # (N, N)
                # choice 1: provide by mmpose
                # push_loss = torch.sum(torch.exp(-diff.pow(2)))
                # choice 2: add variance, given by the paper but the official code does not use it
                # variance = diff.var()
                # push_loss = torch.sum(torch.exp(-diff.pow(2)/(2*variance)))
                # import pdb; pdb.set_trace()
                # choice 3: use the sum of the diff
                diff_sum = diff  # (N, N)
                # print(torch.unique(diff_sum))
                # print(diff_sum.shape)
                push_loss = torch.sum(torch.exp(-diff_sum))
                # print(torch.unique(diff))
                
            # 正则化
            eps = 1e-6
            pull_loss = pull_loss / (N + eps)
            push_loss = push_loss / ((N - 1) * N + eps)
            # import pdb; pdb.set_trace()
            # print(f'pull_loss: {pull_loss}, push_loss: {push_loss}')
        return pull_loss, push_loss 
    
    def forward(self, pred, target, ignore_position):
        '''Forward Function
        Args:
            pred (Tensor): The predicted embedding map. Shape (N, L, H, W). (L is the associated embedding dimension)
            target (Tensor): The target embedding map. Shape (N, 1, H, W).
            ignore_position (Tensor): The mask of ignore position. Shape (N, 1, H, W).
        '''
        # import pdb; pdb.set_trace()
        # dimension_norm = pred.norm(dim=1, keepdim=True)
        # pred = pred / (dimension_norm + 1e-6)  # normalize the embedding vector
        
        bs = pred.size(0)
        L = pred.size(1) # the associated embedding dimension
        
        pull_loss = 0.0
        push_loss = 0.0
        
        for i in range(bs): # calculate the loss for each sample in the batch
            pred_sample = pred[i]
            target_sample = target[i]
            ignore_position_sample = ignore_position[i]
            _pull, _push = self._ae_loss_per_image(pred_sample, target_sample,
                                                   embedding_dim=L, ignore_position=ignore_position_sample)
            # print(f'pull_loss: {_pull}, push_loss: {_push}')
            pull_loss += _pull * self.loss_weight
            push_loss += _push * self.loss_weight * self.push_loss_factor
        # print(f'pull_loss: {pull_loss}, push_loss: {push_loss}')   
        # total_loss = pull_loss + push_loss
         
        # return total_loss
        return pull_loss, push_loss
            
    @property
    def loss_name(self):
        return self._loss_name
