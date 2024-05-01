import polars as pl

def sample(dataframe, fraction):
    return dataframe.group_by("WEEK_NUM") \
                    .map_groups(lambda df: df.group_by("target").map_groups(lambda df: df.sample(fraction=0.1)))