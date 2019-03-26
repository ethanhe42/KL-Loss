# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np

from detectron.core.config import cfg

class CovLossOp(object):
    """
    """

    def __init__(self):
        pass

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: 
                bbox_inw, 
                    N x cls * 4, 
                U, 
                    N x cls * 10
                    0,1,2,3
                      4,5,6
                        7,8
                          9
                bbox_outside_weights
                    N x cls * 4, 
            outputs: loss_bbox
        """
        bbox_inw = inputs[0].data.copy().astype(float)
        N = bbox_inw.shape[0]
        Ncls = bbox_inw.shape[1] / 4
        bbox_inw = bbox_inw.reshape((N, Ncls, 4))
        U = inputs[1].data.copy().reshape((N, Ncls, 10)).astype(float)
        bbox_outside_weights = inputs[2].data.astype(float)
        U_reshape = np.zeros((N, Ncls, 4, 4), dtype=float)

        U_reshape[:,:, 0, 0] = np.exp(U[:, :, 0])
        U_reshape[:,:, 1, 1] = np.exp(U[:, :, 4])
        U_reshape[:,:, 2, 2] = np.exp(U[:, :, 7])
        U_reshape[:,:, 3, 3] = np.exp(U[:, :, 9])
        U_reshape[:,:, 0, 1:]= U[:, :, 1:4]
        U_reshape[:,:, 1, 2:]= U[:, :, 5:7]
        U_reshape[:,:, 2, 3 ]= U[:, :, 8]
        Sigma = np.matmul(np.transpose(U_reshape, (0,1,3,2)), U_reshape)

        loss  = 0.5 * np.matmul(np.matmul(bbox_inw[:, :, None, :], Sigma), bbox_inw[:, :, :, None])
        loss = np.squeeze(loss)
        loss[...] = 0.
        loss -= np.squeeze(U[:,:, 0] + U[:,:, 4] + U[:,:, 7] + U[:,:, 9])
        #loss -= 0.5 * np.squeeze(np.log(np.linalg.det(Sigma))) #  + 1e-5
        loss *= bbox_outside_weights[:, ::4]
        outputs[0].reshape((1,))
        outputs[0].data[...] = np.array([np.sum(loss) / N])

    def backward(self, inputs, outputs):
        """
        Args:
            inputs: bbox_inw, U, bbox_outside_weights
                    loss_bbox
                    d_loss_bbox
            outputs: d_bbox_pred, d_U
        """
        bbox_inw = inputs[0].data.copy().astype(float)
        N = bbox_inw.shape[0]
        Ncls = bbox_inw.shape[1] / 4
        bbox_inw = bbox_inw.reshape((N, Ncls, 4))
        U = inputs[1].data.copy().reshape((N, Ncls, 10)).astype(float)
        bbox_outside_weights = inputs[2].data.astype(float)
        d_loss_bbox = inputs[-1].data.astype(float)
        U_reshape = np.zeros((N, Ncls, 4, 4), dtype=float)
        d_U = np.empty_like(U, dtype=float)
        
        U_reshape[:,:, 0, 0] = np.exp(U[:, :, 0])
        U_reshape[:,:, 1, 1] = np.exp(U[:, :, 4])
        U_reshape[:,:, 2, 2] = np.exp(U[:, :, 7])
        U_reshape[:,:, 3, 3] = np.exp(U[:, :, 9])
        U_reshape[:,:, 0, 1:]= U[:, :, 1:4]
        U_reshape[:,:, 1, 2:]= U[:, :, 5:7]
        U_reshape[:,:, 2, 3 ]= U[:, :, 8]
        Sigma = np.matmul(np.transpose(U_reshape, (0,1,3,2)), U_reshape)

        #d_U_ = np.matmul(np.matmul(U_reshape, bbox_inw[:,:,:,None]), bbox_inw[:,:,None,:]) \
        #        - np.transpose(np.linalg.inv(U_reshape), [0,1,3,2])
        d_U_ = - np.transpose(np.linalg.inv(U_reshape), [0,1,3,2])
        d_U_ *= bbox_outside_weights[:, ::4, None, None]
        d_U[:,:, 0:4] = d_U_[:,:,0,0:]
        d_U[:,:, 4:7] = d_U_[:,:,1,1:] 
        d_U[:,:, 7:9] = d_U_[:,:,2,2:] 
        d_U[:,:,   9] = d_U_[:,:,3,3 ] 
        d_U[:,:,0] *= U_reshape[:,:,0,0]
        d_U[:,:,4] *= U_reshape[:,:,1,1]
        d_U[:,:,7] *= U_reshape[:,:,2,2]
        d_U[:,:,9] *= U_reshape[:,:,3,3]
        d_U = d_U.reshape((N, Ncls*10))

        d_bbox_pred = np.squeeze(np.matmul(bbox_inw[:,:,None,:], Sigma))
        d_bbox_pred *= bbox_outside_weights[:, ::4, None]
        d_bbox_pred = d_bbox_pred.reshape((N, Ncls*4))

        outputs[0].reshape(inputs[0].shape)
        outputs[1].reshape(inputs[1].shape)
        outputs[0].data[...] = d_loss_bbox * d_bbox_pred / N
        outputs[1].data[...] = d_loss_bbox * d_U / N

class UpperTriangularOp(object):
    """
    """

    def __init__(self):
        pass

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: U
                    N x cls * 10
                    0,1,2,3
                      4,5,6
                        7,8
                          9
            outputs: U_reshape
                    N x cls x 4 x 4
        """
        U = inputs[0].data.copy()
        N = U.shape[0]
        Ncls = U.shape[1] / 10
        U = U.reshape((N, Ncls, 10))
        U_reshape = np.zeros((N, Ncls, 4, 4), dtype=U.dtype)

        U_reshape[:,:, 0, 0] = np.exp(U[:, :, 0])
        U_reshape[:,:, 1, 1] = np.exp(U[:, :, 4])
        U_reshape[:,:, 2, 2] = np.exp(U[:, :, 7])
        U_reshape[:,:, 3, 3] = np.exp(U[:, :, 9])
        U_reshape[:,:, 0, 1:]= U[:, :, 1:4]
        U_reshape[:,:, 1, 2:]= U[:, :, 5:7]
        U_reshape[:,:, 2, 3 ]= U[:, :, 8]

        outputs[0].reshape(U_reshape.shape)
        outputs[0].data[...] = U_reshape

    def backward(self, inputs, outputs):
        """
        Args:
            inputs: 
                    U, U_reshape, d_U_reshape
            outputs: d_U
        """
        U = inputs[0].data.copy()
        N = U.shape[0]
        Ncls = U.shape[1] / 10
        U = U.reshape((N, Ncls, 10))
        U_reshape = inputs[1].data.copy()
        d_U_reshape = inputs[2].data.copy()
        d_U = np.empty_like(U)

        d_U[:,:, 0:4] = d_U_reshape[:,:,0,0:]
        d_U[:,:, 4:7] = d_U_reshape[:,:,1,1:] 
        d_U[:,:, 7:9] = d_U_reshape[:,:,2,2:] 
        d_U[:,:,   9] = d_U_reshape[:,:,3,3 ] 
        d_U[:,:,0] *= U_reshape[:,:,0,0]
        d_U[:,:,4] *= U_reshape[:,:,1,1]
        d_U[:,:,7] *= U_reshape[:,:,2,2]
        d_U[:,:,9] *= U_reshape[:,:,3,3]
        d_U = d_U.reshape((N, Ncls*10))

        outputs[0].reshape(inputs[0].shape)
        outputs[0].data[...] = d_U

class LogDetOp(object):
    """Assume that the input is triangular matrix
    """

    def __init__(self):
        pass

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: U
                    N x cls x 10
            outputs: logdet(U)
                    N x cls
        """
        #U = inputs[0].data
        #N = U.shape[0]
        #Ncls = U.shape[1]
        #logdet = np.zeros((N, Ncls), dtype=U.dtype)

        #for i in range(4):
        #    logdet += np.log(U[:, :, i, i])
        #outputs[0].reshape(logdet.shape)
        #outputs[0].data[...] = logdet

        """compact input N x cls x 10"""
        U = inputs[0].data
        
        N = U.shape[0]
        Ncls = U.shape[1]
        logdet = np.zeros((N, Ncls), dtype=U.dtype)

        for i in [0, 4, 7, 9]:
            logdet += U[:, :, i]
        outputs[0].reshape(logdet.shape)
        outputs[0].data[...] = logdet

    def backward(self, inputs, outputs):
        """
        Args:
            inputs: 
                    U, logdet, d_logdet
            outputs: d_U
        """
        #U = inputs[0].data
        #d_logdet = inputs[2].data
        #d_U= np.zeros_like(U)
        #for i in range(4):
        #    d_U[:, :, i, i] = d_logdet / U[:, :, i, i]

        #outputs[0].reshape(d_U.shape)
        #outputs[0].data[...] = d_U

        """compact input N x cls x 10"""
        U = inputs[0].data
        d_logdet = inputs[2].data
        d_U= np.zeros_like(U)
        for i in [0, 4, 7, 9]:
            d_U[:, :, i] = d_logdet
        outputs[0].reshape(d_U.shape)
        outputs[0].data[...] = d_U

class LockOp(object):
    """Lock some components
    """

    def __init__(self):
        pass

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: U
                    N x cls x 10
            outputs: U_out
                    N x cls x 10
        """
        U = inputs[0].data
        U_out = U.copy()
        for i in range(10):
            if i not in [0, 4, 7, 9]:
                U_out[..., i] = 0.
        outputs[0].reshape(U_out.shape)
        outputs[0].data[...] = U_out

    def backward(self, inputs, outputs):
        U = inputs[2].data
        U_out = U.copy()
        for i in range(10):
            if i not in [0, 4, 7, 9]:
                U_out[..., i] = 0.
        outputs[0].reshape(U_out.shape)
        outputs[0].data[...] = U_out

class LogSumExpOp(object):
    """LogSumExp on the last dim
    """

    def __init__(self):
        pass

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: X
                    N x CLS x K
            outputs: Y
                    N x cls
        """
        X = inputs[0].data
        if np.isnan(X).any():
            print(X.shape)
            print(X)
        sumexp = np.exp(X).sum(-1)
        if -np.inf in sumexp or np.inf in sumexp or 0 in sumexp or np.isnan(sumexp).any():
            print(sumexp.shape)
            print(sumexp )
        Y = np.log(sumexp)
        outputs[0].reshape(Y.shape)
        outputs[0].data[...] = Y

    def backward(self, inputs, outputs):
        """
        Args:
            inputs: 
                    X, Y, d_Y
            outputs: d_X
        """
        X = inputs[0].data
        Y = inputs[1].data
        d_Y = inputs[2].data
        
        # d_X = d_Y * exp / sumexp
        #exp(Y) = sumexp
        d_X = d_Y[..., None] * np.exp(X - Y[..., None])

        outputs[0].reshape(d_X.shape)
        outputs[0].data[...] = d_X

class SigmaL1Op(object):
    def __init__(self):
        pass

    def forward(self, inputs, outputs):
        bbox_in = inputs[0].data
        in_abs = np.abs(bbox_in)
        Y = np.zeros_like(bbox_in)
        idxl2 = in_abs <= 1.
        idxl1 = in_abs > 1.
        Y[idxl2] = bbox_in[idxl2]
        Y[idxl1] = np.sign(bbox_in[idxl1]) * np.sqrt((2 * in_abs - 1)[idxl1])
        outputs[0].reshape(Y.shape)
        outputs[0].data[...] = Y

    def backward(self, inputs, outputs):
        """
        Args:
            inputs: 
                    X, Y, d_Y
            outputs: d_X
        """
        bbox_in = inputs[0].data
        in_abs = np.abs(bbox_in)
        idxl2 = in_abs <= 1.
        idxl1 = in_abs > 1.
        Y = inputs[1].data 
        d_Y = inputs[2].data 
        d_X = np.zeros_like(bbox_in)
        d_X[idxl2] = d_Y[idxl2]
        d_X[idxl1] = d_Y[idxl1] * ((2 * in_abs - 1)[idxl1])**-0.5

        outputs[0].reshape(d_X.shape)
        outputs[0].data[...] = d_X

class SimpleOp(object):
    def __init__(self):
        pass

    def forward(self, inputs, outputs):
        outputs[0].reshape(inputs[0].shape)
        outputs[0].data[...] = inputs[0].data

    def backward(self, inputs, outputs):
        outputs[0].reshape(inputs[-1].shape)
        outputs[0].data[...] = inputs[-1].data
