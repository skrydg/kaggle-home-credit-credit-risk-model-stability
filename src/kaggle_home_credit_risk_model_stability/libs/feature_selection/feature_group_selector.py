class FeatureGroupSelector:
    ServiceColumns = "service_columns"
    DateColumns = "date_columns"
    FeatureFractionColumns = "feature_fraction_columns"
    PairwiseDateDiffColumns = "pairwise_date_diff_columns"
    TargetDistributionColumns = "target_distribution_columns"
    MissMatchColumns = "miss_match_columns"
    OneHotEncodingColumns = "one_hot_encoding_columns"
    AgeColumns = "age_columns"
    StdColumns = "std_columns"
    MaxColumns = "max_columns"
    MinColumns = "min_columns"
    MeanColumns = "mean_columns"
    ModeColumns = "mode_columns"
    FirstColumns = "first_columns"
    LastColumns = "last_columns"
    NUniqueColumns = "n_unique_columns"
    AnomalyFeatures = "anomaly_features"

    def __init__(self, include_groups=[], exclude_groups=[]):
        self.include_groups = include_groups
        self.exclude_groups = exclude_groups

    def select(self, column_info, features):
        columns_groups = {
            self.ServiceColumns: ["WEEK_NUM", "case_id", "MONTH", "target", "date_decision"],
            self.DateColumns: [column for column in features if "DATE" in column_info.get_labels(column)],
            self.FeatureFractionColumns: [column for column in features if "MONEY_FRACTION" in column_info.get_labels(column)],
            self.PairwiseDateDiffColumns: [column for column in features if ("DATE_DIFF" in column_info.get_labels(column)) and ("date_decision_diff" not in column)],
            self.TargetDistributionColumns: [column for column in features if ("_target_distribution" in column) or ("_value_persent" in column)],

            self.MissMatchColumns: [column for column in features if "MISMATCH" in column_info.get_labels(column)],
            self.OneHotEncodingColumns: [column for column in features if "ONE_HOT_ENCODING" in column_info.get_labels(column)],
            self.AgeColumns: ["age", "age_bucket"],

            self.StdColumns: [column for column in features if "std_" in column],
            self.MaxColumns: [column for column in features if "max_" in column],
            self.MinColumns: [column for column in features if "min_" in column],
            self.MeanColumns: [column for column in features if "mean_" in column],
            self.ModeColumns: [column for column in features if "mode_" in column],
            self.FirstColumns: [column for column in features if "first_" in column],
            self.LastColumns: [column for column in features if "last_" in column],
            self.NUniqueColumns: [column for column in features if "n_unique" in column],
            self.AnomalyFeatures: [column for column in features if "ANOMALY_FEATURES" in column_info.get_labels(column)],
        }

        assert(len(self.include_groups) > 0) ^ (len(self.exclude_groups) > 0)

        selected_features = []
        if (len(self.include_groups) > 0):
            for group_name in self.include_groups:
                selected_features.extend(columns_groups[group_name])
        else:
            selected_features = features
            for group_name in self.exclude_groups:
                selected_features = [feature for feature in selected_features if feature not in columns_groups[group_name]]
        
        return selected_features