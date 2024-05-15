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
        self.diff_table_for_close = self.get_diff_table_for_close()
        self.diff_table_for_active = self.get_diff_table_for_active()
        self.diff_table = pl.concat([self.diff_table_for_close, self.diff_table_for_active]).group_by("case_id").min()
        
        date_decision_table = self.diff_table
        base_table = self.table_loader.load("base", is_test=self.is_test)
        date_decision_table = date_decision_table.join(base_table, on="case_id", how="left")
        date_decision_table = date_decision_table.with_columns(pl.col("date_decision_diff").fill_null(value=0))
        date_decision_table = date_decision_table.with_columns(pl.col("date_decision").cast(pl.Date) + pl.col("date_decision_diff"))
        
        return date_decision_table[["case_id", "date_decision"]]

    def get_diff_table_for_active(self):

        credit_bureau_a_1 = self.table_loader.load(
            "credit_bureau_a_1",
            columns=["case_id", "dpdmax_139P", "dpdmaxdatemonth_89T", "dpdmaxdateyear_596T", "dateofcredstart_739D", "num_group1"],
            filter = (pl.col("dpdmax_139P") == 0) & (pl.col("dateofcredstart_739D").is_not_null()),
            is_test=self.is_test
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

        diff_table = credit_bureau_a_1[["case_id", "date_decision_diff"]]
        return diff_table

    def get_diff_table_for_close(self):
        credit_bureau_a_1 = self.table_loader.load(
            "credit_bureau_a_1",
            columns=["case_id", "dpdmax_757P", "dpdmaxdatemonth_442T", "dpdmaxdateyear_896T", "dateofcredstart_181D", "num_group1"],
            filter = (pl.col("dpdmax_757P") == 0) & (pl.col("dateofcredstart_181D").is_not_null()),
            is_test=self.is_test
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
        diff_table = credit_bureau_a_1[["case_id", "date_decision_diff"]]
        return diff_table
