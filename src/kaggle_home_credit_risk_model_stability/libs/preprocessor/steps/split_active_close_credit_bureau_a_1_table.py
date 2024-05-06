import numpy as np
import polars as pl


# ideas:
#
# dateofcredend_353D - dateofrealrepmt_138D
#
#
class SplitActiveCloseCreditBureaua1TableStep:
    def __init__(self):
        self.table_name = "credit_bureau_a_1"
        self.service_columns = ["case_id", "num_group1"]

        self.config = {
          "active": {
              "start_column": "dateofcredstart_739D",
              "end_column": "dateofcredend_289D",
              "columns": [
                  #'annualeffectiverate_63L', 'dpdmax_139P', 'nominalrate_281L', 'numberofoutstandinstls_59L', 'numberofoverdueinstlmaxdat_641D', 'numberofoverdueinstls_725L', 'prolongationcount_599L', 'periodicityofpmts_837L'
                  'financialinstitution_591M', 'contractsum_5085717L', 'credlmt_935A', 'dateofcredend_289D', 'dateofcredstart_739D', 'dpdmaxdatemonth_89T', 'instlamount_768A', 'lastupdate_1112D', 'monthlyinstlamount_332A', 'numberofinstls_320L', 'numberofoverdueinstlmax_1039L', 'outstandingamount_362A', 'overdueamount_659A', 'overdueamountmax2_14A', 'overdueamountmax2date_1142D', 'overdueamountmax_155A', 'overdueamountmaxdatemonth_365T', 'purposeofcred_426M', 'residualamount_856A', 'subjectrole_182M', 'totalamount_996A'
              ]
          },
          "close": {
              "start_column": "dateofcredstart_181D",
              "end_column": "dateofcredend_353D",
              "columns": [
                  # 'interestrate_508L', 'annualeffectiverate_199L', 'dateofrealrepmt_138D', 'lastupdate_388D', 'numberofoverdueinstlmaxdat_148D', 'prolongationcount_1120L', 'numberofoverdueinstls_834L', 'periodicityofpmts_1102L', 'outstandingamount_354A', 'residualamount_488A'
                  'financialinstitution_382M', 'credlmt_230A', 'dateofcredend_353D', 'dateofcredstart_181D', 'dpdmax_757P', 'dpdmaxdatemonth_442T', 'instlamount_852A', 'monthlyinstlamount_674A', 'nominalrate_498L', 'numberofinstls_229L', 'numberofoutstandinstls_520L', 'numberofoverdueinstlmax_1151L', 'overdueamount_31A', 'overdueamountmax2_398A', 'overdueamountmax2date_1002D', 'overdueamountmax_35A', 'overdueamountmaxdatemonth_284T', 'purposeofcred_874M', 'totalamount_6A', 'subjectrole_93M'
              ]
          }
        }

        # TODO add other columns
        # self.other_columns = [
        #     'totaldebtoverduevalue_178A', 
        #     'totaldebtoverduevalue_718A',
        #     'totaloutstanddebtvalue_39A',
        #     'totaloutstanddebtvalue_668A',
        #     'debtoutstand_525A',
        #     'debtoverdue_47A',
        #     'numberofcontrsvalue_358L',
        #     'numberofcontrsvalue_258L',
        #     "description_351M",
        #     "refreshdate_3813885D"
        # ]

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)

        for contract_type in self.config.keys():
            dataset, columns_info = self.process_contracts(contract_type, table, dataset, columns_info)
            
        # other
        #dataset.set(f"other_{self.table_name}", table[self.other_columns + self.service_columns])

        dataset.delete(self.table_name)
        return dataset, columns_info
    
    def process_contracts(self, contract_type, table, dataset, columns_info):
        start_column = self.config[contract_type]["start_column"]
        end_column = self.config[contract_type]["end_column"]

        columns = self.config[contract_type]["columns"]

        mask = table[start_column].is_not_null()
        
        table = table.filter(mask)
        table = table[columns + self.service_columns]
        table = table.with_columns((table[end_column].cast(pl.Date) - table[start_column].cast(pl.Date)).dt.total_days().alias("credit_duration"))

        table_name = f"{contract_type}_{self.table_name}"

        dataset.set(table_name, table)
        return dataset, columns_info
