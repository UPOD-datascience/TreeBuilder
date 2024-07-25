import os
import sys
import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from rulefit import RuleFit

from typing import Callable, List
def train_model(model: Callable=None,
                features: List=None,
                target: str="target",
                target_inclusion: List=None,
                train_data: pd.DataFrame=None,
                test_data: pd.DataFrame=None,
                use_class_weights: bool=False):
    return True



def get_reduced_tree(model, train_data, test_data):
    pass