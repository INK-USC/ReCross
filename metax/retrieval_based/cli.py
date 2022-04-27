from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from metax.cli import BaseParser

import os
from metax.models.utils import set_seeds
import sys
import argparse
import logging

import random
import numpy as np
import torch

from metax.models.run_bart import run


def get_parser():
    parser = BaseParser()

    # Used in simple retrieval methods

    parser.add_argument(
        "--do_upstream",
        action='store_true',
        default=False,
        help="do the upstream training",
    )
    parser.add_argument(
        "--do_retrieve",
        action='store_true',
        default=False,
        help="perform adjustments on the model (after upstream) by retrain the model with retrieved subset",
    )

    parser.add_argument(
        "--num_retrieve_round",
        type=int,
        default=10,
        help=
        "select num_retrieve_round different subsets of the whole upstream dataset in retraining stage",
    )
    parser.add_argument(
        "--num_retrieval",
        type=int,
        default=100,
        help="the size of the subset in the retraining stage",
    )

    return parser
