import numpy as np
import polars as pl


class ProcessCreditBureaua1TableStep:
    def __init__(self):
        self.table_name = "credit_bureau_a_1"
        self.service_columns = ["case_id", "num_group1"]

        self.config = {
            "active": {
                "mask_column": "dateofcredstart_739D",
                "financialinstitution_column": "financialinstitution_591M",
                "columns": ['annualeffectiverate_63L', 'contractsum_5085717L', 'credlmt_935A', 'dateofcredend_289D', 'dateofcredstart_739D', 'dpdmax_139P', 'dpdmaxdatemonth_89T', 'instlamount_768A', 'lastupdate_1112D', 'monthlyinstlamount_332A', 'nominalrate_281L', 'numberofinstls_320L', 'numberofoutstandinstls_59L', 'numberofoverdueinstlmax_1039L', 'numberofoverdueinstlmaxdat_641D', 'numberofoverdueinstls_725L', 'outstandingamount_362A', 'overdueamount_659A', 'overdueamountmax2_14A', 'overdueamountmax2date_1142D', 'overdueamountmax_155A', 'overdueamountmaxdatemonth_365T', 'periodicityofpmts_837L', 'prolongationcount_599L', 'purposeofcred_426M', 'residualamount_856A', 'subjectrole_182M', 'totalamount_996A']
            },
            "close": {
                "mask_column": "dateofcredstart_181D",
                "financialinstitution_column": "financialinstitution_382M",
                "columns": ['annualeffectiverate_199L', 'credlmt_230A', 'dateofcredend_353D', 'dateofcredstart_181D', 'dateofrealrepmt_138D', 'dpdmax_757P', 'dpdmaxdatemonth_442T', 'instlamount_852A', 'interestrate_508L', 'lastupdate_388D', 'monthlyinstlamount_674A', 'nominalrate_498L', 'numberofinstls_229L', 'numberofoutstandinstls_520L', 'numberofoverdueinstlmax_1151L', 'numberofoverdueinstlmaxdat_148D', 'numberofoverdueinstls_834L', 'outstandingamount_354A', 'overdueamount_31A', 'overdueamountmax2_398A', 'overdueamountmax2date_1002D', 'overdueamountmax_35A', 'overdueamountmaxdatemonth_284T', 'periodicityofpmts_1102L', 'prolongationcount_1120L', 'purposeofcred_874M', 'residualamount_488A', 'totalamount_6A', 'subjectrole_93M']
            }
        }

        self.finantial_institutions = {
            "active": ['0d39f5db', 'P51_123_23', 'P102_97_118', 'd6a7d943', '50babcd4', 'P150_136_157', 'P133_127_114', 'b619fa46', 'Home Credit', 'P204_66_73'],
            "close": ['dcb42d2c', 'e9bfdb5c', 'a4f0ab55', '952e9882', 'P40_25_35', '50babcd4', '9a93e20f', 'd6a7d943', 'P204_66_73', 'P40_52_135', 'b619fa46', 'P150_136_157', 'P133_127_114', 'Home Credit'],
        }

        self.other_columns = [
            'totaldebtoverduevalue_178A', 
            'totaldebtoverduevalue_718A',
            'totaloutstanddebtvalue_39A',
            'totaloutstanddebtvalue_668A',
            'debtoutstand_525A',
            'debtoverdue_47A',
            'numberofcontrsvalue_358L',
            'numberofcontrsvalue_258L',
            "description_351M",
            "refreshdate_3813885D"
        ]

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)

        for contract_type in self.config.keys():
            dataset = self.process_contract(contract_type, table, dataset)
            
        # other
        dataset.set(f"other_{self.table_name}", table[self.other_columns + self.service_columns])

        dataset.delete(self.table_name)
        return dataset, columns_info
    
    def process_contracts(self, contract_type, table, dataset):
        mask_column = self.config[contract_type]["mask_column"]
        mask = table[mask_column].is_not_null()
        contracts = table.filter(mask)

        financialinstitution_column = self.config[contract_type]["financialinstitution_column"]
        columns = self.config[contract_type]["columns"]

        for finantial_institution in self.finantial_institutions[contract_type]:
            finantial_institution_table = contracts.filter(pl.col(financialinstitution_column) == finantial_institution)
            table_name = f"{contract_type}_{finantial_institution}_{self.table_name}"
            dataset.set(table_name, finantial_institution_table[columns + self.service_columns])
            print(f"Generate new table: {table_name}")

        return dataset
