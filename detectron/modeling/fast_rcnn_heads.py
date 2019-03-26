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

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils

from caffe2.python import workspace, core


import numpy as np

# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    
    model.FC(
        blob_in,
        'bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    if cfg.PRED_STD:
        if cfg.PRED_STD_LOG:
            bias = 0.
            model.FC(
                blob_in, #'blob_in0'
                'bbox_pred_std',
                dim,
                num_bbox_reg_classes * 4,
                weight_init=gauss_fill(0.0001),
                bias_init=const_fill(bias)
            )
            model.net.Copy('bbox_pred_std', 'bbox_pred_std_abs')
            #model.Relu('bbox_pred_std', 'bbox_pred_std_abs')
            #model.net.Sigmoid('bbox_pred_std', 'bbox_pred_std_abs')

def add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.net.ConstantFill([], 'one', value=1., shape=())
    model.net.ConstantFill([], 'half', value=.5, shape=())
    model.net.StopGradient('half', 'half')
    #inside/outside weights are 0,1 matrices
    model.net.StopGradient('bbox_inside_weights', 'bbox_inside_weights')
    model.net.StopGradient('bbox_outside_weights', 'bbox_outside_weights')
    model.net.StopGradient('bbox_targets', 'bbox_targets')
        
    if cfg.PRED_STD:
        ################# bbox_std grad, stop pred
        #log(std)
        #pred0 - y
        model.net.Sub(['bbox_pred', 'bbox_targets'], 'bbox_in')
        #val = in*(pred0 - u)
        model.net.Mul(['bbox_in', 'bbox_inside_weights'], 'bbox_inw')

        #absval
        model.net.Abs('bbox_inw', 'bbox_l1abs')
        # val^2
        model.net.Mul(['bbox_inw', 'bbox_inw'], 'bbox_sq')
        #l12 mask
        model.net.GE(['bbox_l1abs', 'one'], 'wl1', broadcast=1)
        model.net.LT(['bbox_l1abs', 'one'], 'wl2', broadcast=1)
        model.net.Cast('wl1', 'wl1f', to=core.DataType.FLOAT) 
        model.net.Cast('wl2', 'wl2f', to=core.DataType.FLOAT)
        # 0.5 val^2
        model.net.Mul(['bbox_sq', 'wl2f'], 'bbox_l2_')
        model.net.Scale('bbox_l2_', 'bbox_l2', scale=0.5)
        # absval - 0.5
        model.net.Sub(['bbox_l1abs', 'half'], 'bbox_l1abs_', broadcast=1)
        model.net.Mul(['bbox_l1abs_', 'wl1f'], 'bbox_l1')
        # sml1 = w * l1 + w*l2
        model.net.Add(['bbox_l1', 'bbox_l2'], 'bbox_inws')
        #alpha * sml1
        model.net.StopGradient('bbox_inws', 'bbox_inws')
        
        if cfg.PRED_STD_LOG:
            model.net.Scale('bbox_pred_std_abs', 'bbox_pred_std_abs_log', scale=0.5*model.GetLossScale())
            model.net.Negative('bbox_pred_std_abs', 'bbox_pred_std_nabs')
            # e^{-alpha}
            model.net.Exp('bbox_pred_std_nabs', 'bbox_pred_std_nexp')
            model.net.Mul(['bbox_pred_std_nexp', 'bbox_inws'], 'bbox_inws_out')
        else:
            model.net.ConstantFill([], 'sigma', value=0.0001, shape=(1,))
            model.net.Add(['bbox_pred_std_abs', 'sigma'], 'bbox_pred_std_abs_')
            model.net.Log('bbox_pred_std_abs_', 'bbox_pred_std_abs_log_')
            model.net.Scale('bbox_pred_std_abs_log_', 'bbox_pred_std_abs_log', scale=-0.5*model.GetLossScale())
            model.net.Mul(['bbox_pred_std_abs', 'bbox_inws'], 'bbox_inws_out')
        # .5 log(sigma^2)
        model.net.Mul(['bbox_pred_std_abs_log', 'bbox_outside_weights'], 'bbox_pred_std_abs_logw')
        model.net.ReduceMean('bbox_pred_std_abs_logw', 'bbox_pred_std_abs_logwr', axes=[0])
        #bbox_pred grad, stop std
        loss_bbox = model.net.SmoothL1Loss(
            [
                'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
                'bbox_pred_std_abs' if not cfg.PRED_STD_LOG else 'bbox_pred_std_nexp'
            ],
            'loss_bbox',
            scale=model.GetLossScale()
        )
        
        bbox_pred_std_abs_logw_loss = model.net.SumElements(
                'bbox_pred_std_abs_logwr', 'bbox_pred_std_abs_logw_loss')
        model.net.Scale('bbox_inws_out', 'bbox_inws_out', scale=model.GetLossScale())
        model.net.ReduceMean('bbox_inws_out', 'bbox_inws_outr', axes=[0])
        bbox_pred_std_abs_mulw_loss = model.net.SumElements(
                ['bbox_inws_outr'], 'bbox_pred_std_abs_mulw_loss')
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, 
            bbox_pred_std_abs_mulw_loss, 
            bbox_pred_std_abs_logw_loss
            ] + [loss_bbox])
        model.AddLosses(['loss_cls', 
            'bbox_pred_std_abs_mulw_loss', 'bbox_pred_std_abs_logw_loss'
            ] + ['loss_bbox'])
    else:
        loss_bbox = model.net.SmoothL1Loss(
            [
                'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
                'bbox_outside_weights'
            ],
            'loss_bbox',
            scale=model.GetLossScale()
        )

        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
        model.AddLosses(['loss_cls', 'loss_bbox'])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddMetrics('accuracy_cls')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim
