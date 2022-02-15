"""
from https://github.com/chenyuntc/cmr/blob/master/nnutils/laplacian.py by Yun Chen!
Change non-static forward/backward methods to static ones
-------------------------------------------------------------------------------------------------
Computes Lx and it's derivative, where L is the graph laplacian on the mesh with cotangent weights.

1. Given V, F, computes the cotangent matrix (for each face, computes the angles) in pytorch.
2. Then it's taken to NP and sparse L is constructed.

Mesh laplacian computation follows Alec Jacobson's gptoolbox.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import numpy as np
from scipy import sparse


#############
### Utils ###
#############
def convert_as(src, trg):
    return src.to(trg.device).type_as(trg)


class LaplacianLoss(torch.nn.Module):
    def __init__(self, faces, vertices):
        super(LaplacianLoss, self).__init__()
        self.F_np = faces.data.numpy()
        # import pdb; pdb.set_trace()
        self.F = faces.long()  # torch.Tensor(faces).long()#.cuda().long()#.cuda().long() #---for this case#
        self.L = None
        self.vertices = vertices

    def forward(self, V):
        V_np = V.detach().cpu().numpy()
        if self.F.shape[0] != V_np.shape[0]:
            # Recompute laplacian if batch_size doesn't match
            self.L = None
        batchV = V_np.reshape(-1, 3)
        if self.L is None:
            #print("Computing the Laplacian!")
            # Compute cotangents
            verticess = self.vertices.to(device=V.get_device()) #---for this case#
            #print(verticess.shape) -- shape: (778,3)
            sphere_batchV = verticess.unsqueeze(0).repeat(V.shape[0], 1, 1) #---for this case#
            #print(sphere_batchV.shape) -- shape: (25,778,3)
            if self.F.dim() == 2:
                self.F = self.F.unsqueeze(0).repeat(V.shape[0], 1, 1)
            elif self.F.dim() == 3:
                if self.F.shape[0] != V.shape[0]:
                    self.F = self.F[0].unsqueeze(0).repeat(V.shape[0], 1, 1)
            self.F = self.F.to(device=sphere_batchV.device)
            C = cotangent(sphere_batchV.detach(), self.F)
           
            C_np = C.cpu().numpy()
            batchC = C_np.reshape(-1, 3)
            # Adjust face indices to stack:
            offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
            F_np = self.F_np + offset
            batchF = F_np.reshape(-1, 3)

            rows = batchF[:, [1, 2, 0]].reshape(-1)
            cols = batchF[:, [2, 0, 1]].reshape(-1)
            # Final size is BN x BN
            BN = batchV.shape[0]
            L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN, BN))
            L = L + L.T
            # np.sum on sparse is type 'matrix', so convert to np.array
            # import ipdb;ipdb.set_trace()
            M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')

            L = L - M
            # remember this
            self.L = L
        results = Laplacian.apply(V, self.L)
        return results


from torch.autograd.function import once_differentiable


class Laplacian(torch.autograd.Function):

    @staticmethod
    def forward(ctx, V, SL):
        # If forward is explicitly called, V is still a Parameter or Variable
        # But if called through __call__ it's a tensor.
        # This assumes __call__ was used.
        #
        # Input:
        #   V: B x N x 3
        #   F: B x F x 3
        # Outputs: Lx B x N x 3
        #
        # Numpy also doesnt support sparse tensor, so stack along the batch
        
        V_np = V.cpu().numpy()
        batchV = V_np.reshape(-1, 3)
        Lx = SL.dot(batchV).reshape(V_np.shape)
        ctx.L = SL
        out = convert_as(torch.Tensor(Lx), V)
        out = out.to(device=V.device)

        return out#convert_as(torch.Tensor(Lx), V)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = ctx.L.dot(g_o).reshape(grad_out.shape)
        # print('----------------------finish')
        cc = convert_as(torch.Tensor(Lg), grad_out)
        # print(cc.device,'-----')
        return cc, None


def cotangent(V, F):
    # Input:
    #   V: B x N x 3
    #   F: B x F  x3
    # Outputs:
    #   C: B x F x 3 list of cotangents corresponding
    #     angles for triangles, columns correspond to edges 23,31,12

    # B x F x 3 x 3
    indices_repeat = torch.stack([F, F, F], dim=2)

    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0])
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1])
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2])

    l1 = torch.sqrt(((v2 - v3) ** 2).sum(2))
    l2 = torch.sqrt(((v3 - v1) ** 2).sum(2))
    l3 = torch.sqrt(((v1 - v2) ** 2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area
    A = 2 * torch.sqrt(sp * (sp - l1) * (sp - l2) * (sp - l3))

    cot23 = l2 ** 2 + l3 ** 2 - l1 ** 2
    cot31 = l1 ** 2 + l3 ** 2 - l2 ** 2
    cot12 = l1 ** 2 + l2 ** 2 - l3 ** 2

    # 2 in batch
    C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 4

    return C
