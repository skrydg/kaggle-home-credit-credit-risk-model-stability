from collections import defaultdict

class CorrelationGroupsFeatureSelector:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def select(self, dataframe, features):
        numerical_features = [feature for feature in features if dataframe[feature].dtype != pl.Enum]
        categorical_features = [feature for feature in features if dataframe[feature].dtype == pl.Enum]

        null_df = dataframe[numerical_features].select(pl.all().is_null())
        
        null_groups = defaultdict(lambda: list())
        for feature in numerical_features:
            cur_group = null_df[feature].sum()
            null_groups[cur_group].append(feature)
        print("Count nulls:", sorted(list(null_groups.keys())))
        features_to_use = []
        for _, group in null_groups.items():
            if (len(group) == 1):
                features_to_use = features_to_use + group
                continue
            groups_by_correlation = self.group_columns_by_correlation(dataframe, group)            
            assert (np.sum([len(g) for g in groups_by_correlation]) == len(group))
            
            print("groups_by_correlation: ", groups_by_correlation)
            for sub_group in groups_by_correlation:
                features_to_use.append(self.get_most_various_feature(dataframe, sub_group))

        return features_to_use + categorical_features

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