import polars as pl
import numpy as np
import scipy

from collections import defaultdict
from .correlation_groups_getter import CorrelationGroupsGetter

class CorrelationGroupsFeatureSelector:
    def __init__(self, threshold=0.8, kth=0):
        self.correlation_group_getter = CorrelationGroupsGetter(threshold)
        self.kth = kth

    def select(self, dataframe, features):
        selected_feature = []
        correlation_groups = self.correlation_group_getter.get(dataframe, features)
        for group in correlation_groups:
            index = max(0, len(group) - 1 - self.kth)
            selected_feature.append(group[index])
        return selected_feature
