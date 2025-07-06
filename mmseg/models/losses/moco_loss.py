# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class MocoLoss(nn.Module):
    """Associate embedding loss.

    Args:
        loss_weight (float): Weight of AE loss.
        pull_push_weight (float): Weight of pull and push loss.
    """

    def __init__(self, loss_weight=1.0, minimum_instance_pixels=1,
                 T = 0.07, loss_name='loss_moco'):
        super(MocoLoss, self).__init__()
        self.loss_weight = loss_weight
        self.minimum_instance_pixels = minimum_instance_pixels
        self._loss_name = loss_name
        self.T = T
    
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
        # 计算accuracy
        correct_labels = 0
        total_labels = 0

        if N == 0:
            # warnings.warn('No valid instance in the image')
            useless_gradients = pred.mean() 
            moco_loss = torch.tensor(0.0, requires_grad=True).to(pred.device) * useless_gradients
           
        else:
            if N == 1:
                pull_loss = 0.0
                for kpt_embedding, tag in zip(instance_kpt_embeddings, instance_tags):
                    kpt_embedding = kpt_embedding.permute(1, 0)  # (M, L), M is the number of pixels in the instance
                    tag = tag.unsqueeze(0)  # (1, L)
                    pull_loss += self._calculate_rec_loss(kpt_embedding, tag)
            
                # push_loss = torch.tensor(0.0, requires_grad=True).to(pred.device)  # 如果只有一个有效实例或没有有效实例
                moco_loss = pull_loss
                          
            else:
                # TODO:moco loss
                # norm the features in the dimension of L
                normed_instance_kpt_embeddings = []
                normed_instance_tags = []
                for i in range(N):
                    instance_kpt_embedding = instance_kpt_embeddings[i].permute(1, 0)  # (M, L)
                    instance_tag = instance_tags[i].unsqueeze(0)
                    normed_instance_kpt_embedding = instance_kpt_embedding / instance_kpt_embedding.norm(dim=-1, keepdim=True)
                    normed_instance_tag = instance_tag / instance_tag.norm(dim=-1, keepdim=True)
                    normed_instance_kpt_embeddings.append(normed_instance_kpt_embedding)
                    normed_instance_tags.append(normed_instance_tag)
                
                moco_loss = 0.0

                # calculate moco loss
                for i in range(N):
                    q = normed_instance_kpt_embeddings[i] # (M, L)
                    k_pos = normed_instance_tags[i] # (1, L)
                    # negative samples
                    k_neg = []
                    for j in range(N):
                        if j != i:
                            k_neg.append(normed_instance_tags[j])
                    k_neg = torch.cat(k_neg, dim=0) # (N-1, L)
                    logits_pos = torch.einsum('nc,nc ->n', [q,k_pos.detach().repeat(q.size(0),1)]).unsqueeze(-1) # (M, 1)
                    logits_neg = torch.einsum('nc,kc ->nk', [q,k_neg.detach()]) # (M, N-1)

                    logits = torch.cat([logits_pos, logits_neg], dim=1) # (M, N)

                    logits /= self.T

                    # labels
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(pred.device)

                    # calculate acc
                    correct_labels += (logits.argmax(dim=1) == 0).sum().item()
                    total_labels += logits.size(0)

                    moco_loss += F.cross_entropy(logits, labels)
                
                # normalize the loss
                moco_loss = moco_loss / N
        
        return moco_loss, (correct_labels, total_labels)
    
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
    
        loss_total = 0
        correct_all = 0
        total_all = 0

        for i in range(bs): # calculate the loss for each sample in the batch
            pred_sample = pred[i]
            target_sample = target[i]
            ignore_position_sample = ignore_position[i]
            moco_loss_per, (correct, total) = self._ae_loss_per_image(pred_sample, target_sample,
                                                   embedding_dim=L, ignore_position=ignore_position_sample)
            correct_all += correct
            total_all += total

            loss_total += moco_loss_per * self.loss_weight

        acc = correct_all / (total_all + 1e-6)
        # total_loss = pull_loss + push_loss
         
        # return total_loss
        return loss_total, acc
            
    @property
    def loss_name(self):
        return self._loss_name
