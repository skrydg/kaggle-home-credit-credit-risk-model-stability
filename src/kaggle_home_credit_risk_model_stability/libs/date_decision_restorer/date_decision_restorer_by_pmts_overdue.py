import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.table_loader import TableLoader

class DateDecisionRestorerByPmtOverdue:
    def __init__(self, env, is_test=True):
        self.table_loader = TableLoader(env)
        self.is_test = is_test
    
    @staticmethod
    def create_date_from_year_and_month(table, year_column, month_column, date_column):
        table = table.with_columns(table[year_column].cast(pl.Int64).cast(pl.String))
        table = table.with_columns(table[month_column].cast(pl.Int64).cast(pl.String))
        table = table.with_columns((table[year_column] + "-" + table[month_column] + "-15").alias(date_column))
        return table
    
    def restore(self):
        base_table = self.table_loader.load("base", is_test=self.is_test)
        credit_bureau_a_2_terminated = self.table_loader.load(
            "credit_bureau_a_2", 
            columns=[
                "case_id", "num_group1", "num_group2", 
                "pmts_overdue_1152A", "pmts_year_507T", "pmts_month_706T",
            ],
            is_test=self.is_test,
            filter=pl.col("pmts_overdue_1152A").is_not_null()
        )
        credit_bureau_a_2_terminated = self.create_date_from_year_and_month(credit_bureau_a_2_terminated, "pmts_year_507T", "pmts_month_706T", "pmts_date")

        table_2_terminated = credit_bureau_a_2_terminated \
            [["case_id", "num_group1", "num_group2", "pmts_date"]] \
            .group_by(["case_id", "num_group1"]).max().sort("num_group1")
        
        credit_bureau_a_2_existed = self.table_loader.load(
            "credit_bureau_a_2", 
            columns=[
                "case_id", "num_group1", "num_group2", 
                "pmts_overdue_1140A", "pmts_year_1139T", "pmts_month_158T"
            ],
            is_test=self.is_test,
            filter=pl.col("pmts_overdue_1140A").is_not_null()
        )
        credit_bureau_a_2_existed = credit_bureau_a_2_existed.with_columns(credit_bureau_a_2_existed["pmts_year_1139T"].cast(pl.Int64).cast(pl.String))
        credit_bureau_a_2_existed = credit_bureau_a_2_existed.with_columns(credit_bureau_a_2_existed["pmts_month_158T"].cast(pl.Int64).cast(pl.String))
        credit_bureau_a_2_existed = credit_bureau_a_2_existed.with_columns((credit_bureau_a_2_existed["pmts_year_1139T"] + "-" + credit_bureau_a_2_existed["pmts_month_158T"] + "-15").alias("pmts_date"))
        table_2_existed = credit_bureau_a_2_existed \
            [["case_id", "num_group1", "num_group2", "pmts_date"]] \
            .group_by(["case_id", "num_group1"]).max().sort("num_group1")
        
        active_credit_bureau_a_1 = self.table_loader.load(
            "credit_bureau_a_1", 
            columns=[
                "case_id", "num_group1", 
                "dateofcredstart_181D", "lastupdate_388D",
            ],
            is_test=self.is_test,
            filter=pl.col("dateofcredstart_181D").is_not_null()
        )
        active_credit_bureau_a_1 = active_credit_bureau_a_1.with_columns(active_credit_bureau_a_1["num_group1"].cast(pl.Int64))

        close_credit_bureau_a_1 = self.table_loader.load(
            "credit_bureau_a_1", 
            columns=[
                "case_id", "num_group1", 
                "dateofcredstart_739D", "lastupdate_1112D",
            ],
            is_test=self.is_test,
            filter=pl.col("dateofcredstart_739D").is_not_null()
        )
        close_credit_bureau_a_1 = close_credit_bureau_a_1.with_columns(close_credit_bureau_a_1["num_group1"].cast(pl.Int64))
      
        table_1_active = active_credit_bureau_a_1.sort("num_group1")[["case_id", "num_group1", "lastupdate_388D"]]
        table_1_close = close_credit_bureau_a_1.sort("num_group1")[["case_id", "num_group1", "lastupdate_1112D"]]

        table_active = table_1_active.join(table_2_terminated, on=["case_id", "num_group1"], how="inner")
        table_close = table_1_close.join(table_2_existed, on=["case_id", "num_group1"], how="inner")
                
        self.first_table = table_active.with_columns(
            (table_active["pmts_date"].cast(pl.Date) - table_active["lastupdate_388D"].cast(pl.Date)).dt.total_days().alias("date_decision_diff")
        )[["case_id", "date_decision_diff"]]
        
        self.second_table = table_close.with_columns(
            (table_close["pmts_date"].cast(pl.Date) - table_close["lastupdate_1112D"].cast(pl.Date)).dt.total_days().alias("date_decision_diff")
        )[["case_id", "date_decision_diff"]]
        
        self.diff_table = pl.concat([self.first_table, self.second_table]).group_by("case_id").median()[["case_id", "date_decision_diff"]]
        
        date_decision_table = self.diff_table
        base_table = self.table_loader.load("base", is_test=self.is_test)
        date_decision_table = date_decision_table.join(base_table, on="case_id", how="left")
        date_decision_table = date_decision_table.with_columns(pl.col("date_decision_diff").fill_null(value=0))
        date_decision_table = date_decision_table.with_columns(pl.col("date_decision").cast(pl.Date) + pl.col("date_decision_diff"))
        
        return date_decision_table[["case_id", "date_decision"]]