import polars as pl
import numpy as np

from kaggle_home_credit_risk_model_stability.libs.input.table_loader import TableLoader

class DateDecisionRestorerByDpDMaxDate:
    def __init__(self, env, is_test=True):
        self.table_loader = TableLoader(env)
        self.is_test = is_test
    
    @staticmethod
    def create_date_from_year_month_day(table, year_column, month_column, day_column, date_column):
        table = table.with_columns(table[year_column].cast(pl.Int64).cast(pl.String))
        table = table.with_columns(table[month_column].cast(pl.Int64).cast(pl.String))

        day = table[day_column].cast(pl.Int64).fill_null(value=15)
        day = np.minimum(day.to_numpy(), 28)

        table = table.with_columns(pl.Series(day).alias(day_column).cast(pl.Int64).cast(pl.String))

        table = table.with_columns((table[year_column] + "-" + table[month_column] + "-" + table[day_column]).alias(date_column))
        return table
    
    @staticmethod
    def column_to_date(table, columns):
      return table.with_columns(table[columns].cast(pl.Date))

    def restore(self):
        return self.restore_for_close()

    def restore_for_active(self):
        base_table = self.table_loader.load("base")
        credit_bureau_a_1 = self.table_loader.load(
            "credit_bureau_a_1",
            columns=["case_id", "dpdmax_139P", "dpdmaxdatemonth_89T", "dpdmaxdateyear_596T", "dateofcredstart_739D", "num_group1"],
            filter = (pl.col("dpdmax_139P") == 0) & (pl.col("dateofcredstart_739D").is_not_null())
        )

        credit_bureau_a_1 = self.column_to_date(credit_bureau_a_1, "dateofcredstart_739D")
        credit_bureau_a_1 = credit_bureau_a_1.with_columns(credit_bureau_a_1["dateofcredstart_739D"].dt.day().alias("dpdmaxday"))

        credit_bureau_a_1 = self.create_date_from_year_month_day(credit_bureau_a_1, "dpdmaxdateyear_596T", "dpdmaxdatemonth_89T", "dpdmaxday", "dpdmaxdate")
        credit_bureau_a_1 = self.column_to_date(credit_bureau_a_1, "dpdmaxdate")

        credit_bureau_a_1 = credit_bureau_a_1[["dateofcredstart_739D", "dpdmaxdate", "num_group1", "case_id"]]
        credit_bureau_a_1 = credit_bureau_a_1.with_columns(
            pl.when(credit_bureau_a_1["dpdmaxdate"].dt.day() >= 15)
              .then(credit_bureau_a_1["dpdmaxdate"].dt.offset_by("-1mo"))
              .otherwise(credit_bureau_a_1["dpdmaxdate"]))

        credit_bureau_a_1 = credit_bureau_a_1.sort(["case_id", "num_group1"]).group_by("case_id").last()

        credit_bureau_a_1 = credit_bureau_a_1.with_columns(
            (pl.col("dpdmaxdate") - pl.col("dateofcredstart_739D")).dt.total_days().alias("date_decision_diff")
        )
        self.table = credit_bureau_a_1
        self.diff_table = credit_bureau_a_1[["case_id", "date_decision_diff"]]

        base_table = base_table.join(self.diff_table, on="case_id", how="left")
        base_table = base_table.with_columns(pl.col("date_decision_diff").fill_null(value=0))
        base_table = base_table.with_columns(pl.col("date_decision").cast(pl.Date) + pl.col("date_decision_diff"))
        
        return base_table[["case_id", "date_decision"]]
    
    def restore_for_close(self):
        base_table = self.table_loader.load("base")
        credit_bureau_a_1 = self.table_loader.load(
            "credit_bureau_a_1",
            columns=["case_id", "dpdmax_757P", "dpdmaxdatemonth_442T", "dpdmaxdateyear_896T", "dateofcredstart_181D", "num_group1"],
            filter = (pl.col("dpdmax_757P") == 0) & (pl.col("dateofcredstart_181D").is_not_null())
        )

        credit_bureau_a_1 = self.column_to_date(credit_bureau_a_1, "dateofcredstart_181D")
        credit_bureau_a_1 = credit_bureau_a_1.with_columns(credit_bureau_a_1["dateofcredstart_181D"].dt.day().alias("dpdmaxday"))

        credit_bureau_a_1 = self.create_date_from_year_month_day(credit_bureau_a_1, "dpdmaxdateyear_896T", "dpdmaxdatemonth_442T", "dpdmaxday", "dpdmaxdate")
        credit_bureau_a_1 = self.column_to_date(credit_bureau_a_1, "dpdmaxdate")

        credit_bureau_a_1 = credit_bureau_a_1[["dateofcredstart_181D", "dpdmaxdate", "num_group1", "case_id"]]
        credit_bureau_a_1 = credit_bureau_a_1.with_columns(
            pl.when(credit_bureau_a_1["dpdmaxdate"].dt.day() >= 15)
              .then(credit_bureau_a_1["dpdmaxdate"].dt.offset_by("-1mo"))
              .otherwise(credit_bureau_a_1["dpdmaxdate"]))

        credit_bureau_a_1 = credit_bureau_a_1.sort(["case_id", "num_group1"]).group_by("case_id").last()

        credit_bureau_a_1 = credit_bureau_a_1.with_columns(
            (pl.col("dpdmaxdate") - pl.col("dateofcredstart_181D")).dt.total_days().alias("date_decision_diff")
        )
        self.diff_table = credit_bureau_a_1[["case_id", "date_decision_diff"]]

        base_table = base_table.join(self.diff_table, on="case_id", how="left")
        base_table = base_table.with_columns(pl.col("date_decision_diff").fill_null(value=0))
        base_table = base_table.with_columns(pl.col("date_decision").cast(pl.Date) + pl.col("date_decision_diff"))
        
        return base_table[["case_id", "date_decision"]]