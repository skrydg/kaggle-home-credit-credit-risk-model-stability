import numpy as np
import polars as pl


# ideas:
#
# dateofcredend_353D - dateofrealrepmt_138D
#
#
class ProcessCreditBureaua1TableStep:
    def __init__(self, config = None, finantial_institutions = None):
        self.table_name = "credit_bureau_a_1"
        self.service_columns = ["case_id", "num_group1"]

        if config is None:
          config = {
            "active": {
                "mask_column": "dateofcredstart_739D",
                "financialinstitution_column": "financialinstitution_591M",
                "columns": [
                    #'annualeffectiverate_63L', 'dpdmax_139P', 'nominalrate_281L', 'numberofoutstandinstls_59L', 'numberofoverdueinstlmaxdat_641D', 'numberofoverdueinstls_725L', 'prolongationcount_599L', 
                    'contractsum_5085717L', 'credlmt_935A', 'dateofcredend_289D', 'dateofcredstart_739D', 'dpdmaxdatemonth_89T', 'instlamount_768A', 'lastupdate_1112D', 'monthlyinstlamount_332A', 'numberofinstls_320L', 'numberofoverdueinstlmax_1039L', 'outstandingamount_362A', 'overdueamount_659A', 'overdueamountmax2_14A', 'overdueamountmax2date_1142D', 'overdueamountmax_155A', 'overdueamountmaxdatemonth_365T', 'periodicityofpmts_837L', 'purposeofcred_426M', 'residualamount_856A', 'subjectrole_182M', 'totalamount_996A'
                ]
            },
            "close": {
                "mask_column": "dateofcredstart_181D",
                "financialinstitution_column": "financialinstitution_382M",
                "columns": [
                    # 'interestrate_508L', 'annualeffectiverate_199L', 'dateofrealrepmt_138D', 'lastupdate_388D', 'numberofoverdueinstlmaxdat_148D', 'prolongationcount_1120L',
                    'credlmt_230A', 'dateofcredend_353D', 'dateofcredstart_181D', 'dpdmax_757P', 'dpdmaxdatemonth_442T', 'instlamount_852A', 'monthlyinstlamount_674A', 'nominalrate_498L', 'numberofinstls_229L', 'numberofoutstandinstls_520L', 'numberofoverdueinstlmax_1151L', 'numberofoverdueinstls_834L', 'outstandingamount_354A', 'overdueamount_31A', 'overdueamountmax2_398A', 'overdueamountmax2date_1002D', 'overdueamountmax_35A', 'overdueamountmaxdatemonth_284T', 'periodicityofpmts_1102L', 'purposeofcred_874M', 'residualamount_488A', 'totalamount_6A', 'subjectrole_93M'
                ]
            }
          }

        if finantial_institutions is None:
          finantial_institutions = {
              "active": ['P150_136_157', 'P133_127_114', 'b619fa46', 'Home Credit', 'P204_66_73'],
              "close": ['P40_52_135', 'b619fa46', 'P150_136_157', 'P133_127_114', 'Home Credit']
          }

        self.config = config
        self.finantial_institutions = finantial_institutions

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
        mask_column = self.config[contract_type]["mask_column"]
        mask = table[mask_column].is_not_null()
        contracts = table.filter(mask)

        financialinstitution_column = self.config[contract_type]["financialinstitution_column"]
        columns = self.config[contract_type]["columns"]

        for finantial_institution in self.finantial_institutions[contract_type]:
            finantial_institution_table = contracts.filter(pl.col(financialinstitution_column) == finantial_institution)
            finantial_institution_table = finantial_institution_table[columns + self.service_columns]

            table_name = f"{contract_type}_{finantial_institution}_{self.table_name}"

            for column in columns:
                new_column_name = f"{column}_{table_name}"
                labels = columns_info.get_labels(column)
                columns_info.add_labels(new_column_name, labels)

            finantial_institution_table = finantial_institution_table.rename({column: f"{column}_{table_name}" for column in columns})
            dataset.set(table_name, finantial_institution_table)

        print(f"Generate {len(self.finantial_institutions[contract_type])} tables for contract_type={contract_type}")

        return dataset, columns_info
