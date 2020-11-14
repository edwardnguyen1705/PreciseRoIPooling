# -*- coding: utf-8 -*-
# File   : test_prroi_pooling2d.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 18/02/2018
#
# This file is part of Jacinle.
# $ export PYTHONPATH="$PWD"

import sys
sys.path.insert(0, "./")

import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtestcase as ttc # https://github.com/phohenecker/torch-test-case
from prroi_pool import PrRoIPool2D
from torch.autograd import Variable
from torch.nn.utils import rnn


def as_numpy(v):
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()


class TestPrRoIPool2D(unittest.TestCase):
    def setUp(self):
        self.test_case = ttc.TorchTestCase()

    def test_forward(self):
        pool = PrRoIPool2D(7, 7, spatial_scale=0.5)
        features = torch.rand((4, 16, 24, 32)).cuda()
        rois = torch.tensor([
            [0, 0, 0, 14, 14],
            [1, 14, 14, 28, 28],
        ]).float().cuda()

        out = pool(features, rois)
        out_gold = F.avg_pool2d(features, kernel_size=2, stride=1)

        self.assertTensorClose(out, torch.stack((
            out_gold[0, :, :7, :7],
            out_gold[1, :, 7:14, 7:14],
        ), dim=0))

    def assertTensorClose(self, a, b, atol=1e-3, rtol=1e-3):
        npa, npb = as_numpy(a), as_numpy(b)
        self.assertTrue(
                np.allclose(npa, npb, atol=atol),
                'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())
        )


    def test_backward_shapeonly(self):
        pool = PrRoIPool2D(2, 2, spatial_scale=0.5)

        features = torch.rand((4, 2, 24, 32)).cuda()
        rois = torch.tensor([
            [0, 0, 0, 4, 4],
            [1, 14, 14, 18, 18],
        ]).float().cuda()
        features.requires_grad = rois.requires_grad = True
        out = pool(features, rois)

        loss = out.sum()
        loss.backward()

        self.test_case.assertEqual(features.size(), features.grad.size())
        self.test_case.assertEqual(rois.size(), rois.grad.size())
 

if __name__ == '__main__':
    unittest.main()
