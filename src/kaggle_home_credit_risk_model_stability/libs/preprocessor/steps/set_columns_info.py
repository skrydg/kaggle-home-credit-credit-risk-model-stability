import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset
from kaggle_home_credit_risk_model_stability.libs.preprocessor import ColumnsInfo

class SetColumnsInfoStep:
    def __init__(self):
        self.columns_info = None
        
    def process_train_dataset(self, train_dataset_generator):
        train_dataset, raw_tables_info = next(train_dataset_generator)
        self.columns_info = ColumnsInfo(raw_tables_info)
        yield self.set_columns_info(train_dataset, self.columns_info)

        for train_dataset, _ in train_dataset_generator:
            yield train_dataset, self.columns_info
    
    def set_columns_info(self, train_dataset, columns_info):
        for name, table in train_dataset.get_tables():
              for column in table.columns:
                  columns_info.set_table_name(column, name)
                  columns_info.add_label(column, "RAW")
                  if column in ("WEEK_NUM", "case_id", "MONTH", "num_group1", "num_group2", "target"):
                      columns_info.add_label(column, "SERVICE")
                      continue

                  if (column[-1] == "D") or (column in ["date_decision"]):
                      columns_info.add_label(column, "DATE")
                  elif (column[-1] in ["M"]) or (table[column].dtype == pl.String):
                      columns_info.add_label(column, "CATEGORICAL")
                      
                  if column in ["mainoccupationinc_384A", "contractssum_5085716L", "pmtssum_45A", "amtinstpaidbefduel24m_4187115A", "annuity_780A", "annuitynextmonth_57A", "avginstallast24m_3658937A", "avginstallast24m_3658937A", "avglnamtstart24m_4525187A", "avgoutstandbalancel6m_4187114A", "avgoutstandbalancel6m_4187114A", "avgpmtlast12m_4525200A", "credamount_770A", "currdebt_22A", "currdebtcredtyperange_828A", "disbursedcredamount_1113A", "disbursedcredamount_1113A", "downpmt_116A", "inittransactionamount_650A", "lastapprcredamount_781A", "lastotherinc_902A", "lastotherlnsexpense_631A", "lastrejectcredamount_222A", "maininc_215A", "maxannuity_159A", "maxannuity_4075009A", "maxdebt4_972A", "maxinstallast24m_3658928A", "maxlnamtstart6m_4525199A", "maxoutstandbalancel12m_4187113A", "maxpmtlast3m_4525190A", "price_1097A", "sumoutstandtotal_3546847A", "sumoutstandtotalest_4493215A", "totaldebt_9A", "totalsettled_863A", "totinstallast1m_4525188A", 'annuity_853A', 'byoccupationinc_3656910L', 'credacc_actualbalance_314A', 'credacc_credlmt_575A', 'credamount_590A', 'currdebt_94A', 'downpmt_134A', 'mainoccupationinc_437A', 'outstandingdebt_522A', "amount_4527230A", "amount_4917619A", "pmtamount_36A", 'amount_1115A', 'credlmt_1052A', 'credlmt_228A', 'credlmt_3940954A', 'debtpastduevalue_732A', 'debtvalue_227A', 'installmentamount_644A', 'installmentamount_833A', 'instlamount_892A', 'residualamount_127A', 'residualamount_3940956A', 'totalamount_503A', 'totalamount_881A', 'amtdebitincoming_4809443A', 'amtdebitoutgoing_4809440A', 'amtdepositbalance_4809441A', 'amtdepositincoming_4809444A', 'amtdepositoutgoing_4809442A', 'amount_416A', 'last180dayaveragebalance_704A', 'last180dayturnover_1134A', 'last30dayturnover_651A', \
                                "contractsum_5085717L", "credlmt_230A", "credlmt_935A", "debtoutstand_525A", "debtoverdue_47A", "instlamount_768A", "instlamount_852A", "monthlyinstlamount_332A", "monthlyinstlamount_674A", "outstandingamount_354A", "outstandingamount_362A", "overdueamount_31A", "overdueamount_659A", "overdueamountmax2_14A", "overdueamountmax2_398A", "overdueamountmax2date_1002D", "overdueamountmax2date_1142D", "overdueamountmax_155A", "overdueamountmax_35A", "residualamount_488A", "residualamount_856A", "totalamount_6A", "totalamount_996A", "totaldebtoverduevalue_178A", "totaldebtoverduevalue_718A", "totaloutstanddebtvalue_39A", "totaloutstanddebtvalue_668A", \
                                  "pmts_overdue_1140A", "pmts_overdue_1152A"]:
                      columns_info.add_label(column, "MONEY")

        return train_dataset, columns_info

    def process_test_dataset(self, test_dataset_generator):
        for test_dataset, _ in test_dataset_generator:
            yield test_dataset, self.columns_info