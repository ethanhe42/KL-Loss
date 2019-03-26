from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import six
from cov_loss import *
import unittest

from caffe2.python import core, workspace
from caffe2.python.core import CreatePythonOperator
import caffe2.python.hypothesis_test_util as hu
from hypothesis import given
import hypothesis.strategies as st
from caffe2.proto import caffe2_pb2
from caffe2.python import gradient_checker

import detectron.utils.c2 as c2_utils
import detectron.utils.logging as logging_utils

#class PythonOpTest(hu.HypothesisTestCase):
#    @given(inputs=hu.tensors(n=3), **hu.gcs)
#    def test_gradient_multiple_with_indices(self, inputs, gc, dc):
#        (x1, x2, x3) = inputs
#
#        op = CreatePythonOperator(
#            CovLossOp().forward, ["x1", "x2", "x3"], ["y1"],
#            grad_f=CovLossOp().backward,
#            grad_input_indices=[0, 1]
#            )
#
#        self.assertGradientChecks(gc, op, [x1, x2, x3], 0, [0])
#        self.assertDeviceChecks(dc, op, [x1, x2, x3], [0, 1, 2])


class CovLossTest(unittest.TestCase):
    def test_forward_and_gradient(self):
        N = 64
        Ncls = 2
        Y = np.random.randn(N, 4 * Ncls).astype(np.float32)
        Y_hat = np.random.randn(N, 4 * Ncls).astype(np.float32)
        U = np.random.randn(N, 10 * Ncls).astype(np.float32)
        print(np.prod(Y.shape))
        print(np.prod(U.shape))
        inside_weights = np.random.randn(N, 4 * Ncls).astype(np.float32)
        inside_weights[inside_weights < 0] = 0
        #outside_weights = np.ones((N, 4 * Ncls)).astype(np.float32)
        outside_weights_ind = np.random.randint(0, Ncls, N)
        outside_weights = np.zeros((N, 4 * Ncls)).astype(np.float32)
        for i in range(N):
            ind = 4*outside_weights_ind[i]
            outside_weights[i, ind:ind+4] = 1
        scale = np.random.random()
        beta = np.random.random()
        bbox_inw = outside_weights * (Y - Y_hat)

        #op = CreatePythonOperator(
        #    CovLossOp().forward, ["x1", "x2", "x3"], ["y1"],
        #    grad_f=CovLossOp().backward,
        #    grad_input_indices=[0, 1]
        #    )

        #op = CreatePythonOperator(
        #    UpperTriangularOp().forward, ["x1"], ["y1"],
        #    grad_f=UpperTriangularOp().backward,
        #    #grad_input_indices=[0, 1]
        #    )
        op = CreatePythonOperator(
            LogDetOp().forward, ["x1"], ["y1"],
            grad_f=LogDetOp().backward,
            #grad_input_indices=[0, 1]
            )
        #op = CreatePythonOperator(
        #    SimpleOp().forward, ["x1"], ["y1"],
        #    grad_f=SimpleOp().backward,
        #    #grad_input_indices=[0, 1]
        #    )

        gc = gradient_checker.GradientChecker(
            stepsize=0.005,
            threshold=0.005,
            #device_option=core.DeviceOption(caffe2_pb2.CUDA, 0)
        )

        #res, grad, grad_estimated = gc.CheckSimple(
        #    op, [bbox_inw, U, outside_weights], 0, [0]
        #)
        U = U.reshape((N, Ncls, 10))
        U_reshape = np.zeros((N, Ncls, 4, 4), dtype=U.dtype)
        U_reshape[:,:, 0, 0] = np.exp(U[:, :, 0])
        U_reshape[:,:, 1, 1] = np.exp(U[:, :, 4])
        U_reshape[:,:, 2, 2] = np.exp(U[:, :, 7])
        U_reshape[:,:, 3, 3] = np.exp(U[:, :, 9])
        U_reshape[:,:, 0, 1:]= U[:, :, 1:4]
        U_reshape[:,:, 1, 2:]= U[:, :, 5:7]
        U_reshape[:,:, 2, 3 ]= U[:, :, 8]
        res, grad, grad_estimated = gc.CheckSimple(
            op, [U_reshape], 0, [0]
        )
        #res, grad, grad_estimated = gc.CheckSimple(
        #    op, [U], 0, [0]
        #)

        self.assertTrue(
            grad.shape == grad_estimated.shape,
            'Fail check: grad.shape != grad_estimated.shape'
        )

        ## To inspect the gradient and estimated gradient:
        #np.set_printoptions(precision=3, suppress=True)
        #print('grad:')
        #print(grad)
        #print('grad_estimated:')
        #print(grad_estimated)

        self.assertTrue(res)

if __name__ == '__main__':
    #c2_utils.import_detectron_ops()
    #assert 'SmoothL1Loss' in workspace.RegisteredOperators()
    logging_utils.setup_logging(__name__)
    unittest.main()
