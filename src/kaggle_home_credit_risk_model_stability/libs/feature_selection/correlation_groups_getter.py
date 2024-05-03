import polars as pl
import numpy as np
import scipy

from collections import defaultdict

class CorrelationGroupsGetter:
    def __init__(self, numerical_threshold=0.8, categorical_threashold=0.9):
        self.numerical_threshold = numerical_threshold
        self.categorical_threashold = categorical_threashold

    def get(self, dataframe, features):
        features = list(sorted(features))
        numerical_features = [feature for feature in features if dataframe[feature].dtype != pl.Enum]
        categorical_features = [feature for feature in features if dataframe[feature].dtype == pl.Enum]

        return self.get_impl(dataframe, numerical_features, CorrelationGroupsGetter.get_correlation_for_numerical_features, self.numerical_threshold) + \
              self.get_impl(dataframe, categorical_features, CorrelationGroupsGetter.get_correlation_for_categorical_features, self.categorical_threashold)
    
    def get_impl(self, dataframe, features, pairwise_correlation_getter, threashold):
        null_df = dataframe[features].select(pl.all().is_null())
        null_df = null_df.sum()
        null_array = sorted(list(zip(null_df.to_numpy().tolist()[0], null_df.columns)))

        null_groups = defaultdict(lambda: list())
        for count_null, feature in null_array:
            cur_group = count_null
            for existing_group in null_groups.keys():
                if abs((existing_group - cur_group)) / max(1, (existing_group + cur_group)) < 0.01:
                    cur_group = existing_group
                    break
            null_groups[cur_group].append(feature)

        print("Count nulls in features:", sorted(list(null_groups.keys())))
        correlation_groups = []
        for _, group in null_groups.items():
            groups_by_correlation = self.group_columns_by_correlation(dataframe, group, pairwise_correlation_getter, threashold)            
            assert (np.sum([len(g) for g in groups_by_correlation]) == len(group))
            
            print("groups_by_correlation in features: ", groups_by_correlation)
            for sub_group in groups_by_correlation:
                correlation_groups.append(self.sort_group(dataframe, sub_group))

        return correlation_groups

    def group_columns_by_correlation(self, dataframe, features, pairwise_correlation_getter, threashold):
        groups = []
        remaining_features = features
        while len(remaining_features) > 0:
            feature = remaining_features[0]
            group = []
            for cur_feature in remaining_features:
                if (cur_feature == feature) or (pairwise_correlation_getter(dataframe, feature, cur_feature) >= threashold):
                    group.append(cur_feature)
            groups.append(group)
            remaining_features = [feature for feature in remaining_features if feature not in group]
        
        return groups

    def sort_group(self, dataframe, features):
        features = list(reversed(features)) # temporary for back compatability
        sorted_index = np.argsort(dataframe[features].select(pl.all().n_unique()).to_numpy()[0])
        return np.array(features)[sorted_index].tolist()
        
    @staticmethod
    def get_correlation_for_numerical_features(dataframe, feature1, feature2):
        return dataframe.select(pl.corr(feature1, feature2))[0, 0]
    
    @staticmethod
    def get_correlation_for_categorical_features(dataframe, feature1, feature2):
        cur_df = dataframe[[feature1, feature2]].with_columns(pl.lit(1).alias("const_1"))

        pivot_df = cur_df.pivot(index=feature1, columns=feature2, values="const_1", aggregate_function="sum")
        pivot_df = pivot_df.fill_null(0)
        pivot_df = pivot_df.drop(feature1)

        return scipy.stats.contingency.association(pivot_df.to_numpy(), method='pearson')

