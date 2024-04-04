import polars as pl
import numpy as np
import scipy

from collections import defaultdict

class CorrelationGroupsFeatureSelector:
    def __init__(self, columns_info, threshold=0.8):
        self.columns_info = columns_info
        self.threshold = threshold

    def select(self, dataframe, features):
        numerical_features = [feature for feature in features if dataframe[feature].dtype != pl.Enum]
        categorical_features = [feature for feature in features if dataframe[feature].dtype == pl.Enum]

        return self.select_numerical(dataframe, numerical_features) + \
              self.select_categorical(dataframe, categorical_features)
    
    def select_categorical(self, dataframe, categorical_features):
        bad_mask = np.zeros(len(categorical_features), dtype=bool)
        for feature1_index in range(len(categorical_features)):
            for feature2_index in range(feature1_index + 1, len(categorical_features)):
                feature1 = categorical_features[feature1_index]
                feature2 = categorical_features[feature2_index]

                # Skip features if they part of the same categorical feature
                if ("PART" in self.columns_info.get_labels(feature1)) and \
                    ("PART" in self.columns_info.get_labels(feature2)) and \
                    (self.columns_info.get_ancestor(feature1) == self.columns_info.get_ancestor(feature2)):
                    continue

                corr_coef = self.get_correlation_for_categorical_features(dataframe, feature1, feature2)
                if (corr_coef > self.threshold):
                    print(f"Categorical feature with high correlation, feature1={feature1}, feature2={feature2}")
                    if (dataframe[feature1].n_unique() >= dataframe[feature2].n_unique()):
                        bad_mask[feature2_index] = True
        return np.array(categorical_features)[~bad_mask].tolist()
    
    def get_correlation_for_categorical_features(self, dataframe, feature1, feature2):
        cur_df = dataframe[[feature1, feature2]].with_columns(pl.lit(1).alias("const_1"))

        pivot_df = cur_df.pivot(index=feature1, columns=feature2, values="const_1", aggregate_function="sum")
        pivot_df = pivot_df.fill_null(0)
        pivot_df = pivot_df.drop(feature1)

        return scipy.stats.contingency.association(pivot_df.to_numpy(), method='pearson')

    def select_numerical(self, dataframe, numerical_features):
        null_df = dataframe[numerical_features].select(pl.all().is_null())
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

        print("Count nulls in numerical features:", sorted(list(null_groups.keys())))
        features_to_use = []
        for _, group in null_groups.items():
            if (len(group) == 1):
                features_to_use = features_to_use + group
                continue
            groups_by_correlation = self.group_columns_by_correlation(dataframe, group)            
            assert (np.sum([len(g) for g in groups_by_correlation]) == len(group))
            
            print("groups_by_correlation in numerical features: ", groups_by_correlation)
            for sub_group in groups_by_correlation:
                features_to_use.append(self.get_most_various_feature(dataframe, sub_group))

        return features_to_use

    def group_columns_by_correlation(self, dataframe, features):
        groups = []
        remaining_features = features
        while len(remaining_features) > 0:
            feature = remaining_features[0]
            group = []
            for cur_feature in remaining_features:
                if (cur_feature == feature) or (self.get_correlation(dataframe, feature, cur_feature) >= self.threshold):
                    group.append(cur_feature)
            groups.append(group)
            remaining_features = [feature for feature in remaining_features if feature not in group]
        
        return groups

    def get_most_various_feature(self, dataframe, features):
        index = np.argmax(dataframe[features].select(pl.all().n_unique()).to_numpy()[0])
        return features[index]
        
    def get_correlation(self, dataframe, feature1, feature2):
        return dataframe.select(pl.corr(feature1, feature2))[0, 0]
