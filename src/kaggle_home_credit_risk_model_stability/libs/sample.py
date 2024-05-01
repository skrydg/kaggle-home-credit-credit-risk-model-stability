import polars as pl

def sample(dataframe, false_target_fraction, true_target_fraction):
    return dataframe.group_by("WEEK_NUM").map_groups(
              lambda df: pl.concat([
                    df.filter(pl.col("target") == 0).sample(fraction = false_target_fraction),
                    df.filter(pl.col("target") == 1).sample(fraction = true_target_fraction)
                ], how="vertical_relaxed")
            )