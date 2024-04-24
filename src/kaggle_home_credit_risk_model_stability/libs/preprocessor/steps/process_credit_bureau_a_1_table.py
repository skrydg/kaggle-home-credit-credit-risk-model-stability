import numpy as np
import polars as pl


class ProcessCreditBureaua1TableStep:
    def __init__(self):
        self.table_name = "credit_bureau_a_1"
        self.service_columns = ["case_id", "num_group1"]
        self.close_contracts_columns = ['annualeffectiverate_199L', 'classificationofcontr_400M', 'contractst_964M', 'credlmt_230A', 'dateofcredend_353D', 'dateofcredstart_181D', 'dateofrealrepmt_138D', 'dpdmax_757P', 'dpdmaxdatemonth_442T', 'financialinstitution_382M', 'instlamount_852A', 'interestrate_508L', 'lastupdate_388D', 'monthlyinstlamount_674A', 'nominalrate_498L', 'numberofinstls_229L', 'numberofoutstandinstls_520L', 'numberofoverdueinstlmax_1151L', 'numberofoverdueinstlmaxdat_148D', 'numberofoverdueinstls_834L', 'outstandingamount_354A', 'overdueamount_31A', 'overdueamountmax2_398A', 'overdueamountmax2date_1002D', 'overdueamountmax_35A', 'overdueamountmaxdatemonth_284T', 'periodicityofpmts_1102L', 'prolongationcount_1120L', 'purposeofcred_874M', 'residualamount_488A', 'totalamount_6A', 'subjectrole_93M']
        self.close_contracts_columns.remove("financialinstitution_382M")
        self.close_contracts_columns.remove("classificationofcontr_400M")
        self.close_contracts_columns.remove("contractst_964M")
        self.active_contracts_columns = ['annualeffectiverate_63L', 'classificationofcontr_13M', 'contractst_545M', 'contractsum_5085717L', 'credlmt_935A', 'dateofcredend_289D', 'dateofcredstart_739D', 'dpdmax_139P', 'dpdmaxdatemonth_89T', 'financialinstitution_591M', 'instlamount_768A', 'lastupdate_1112D', 'monthlyinstlamount_332A', 'nominalrate_281L', 'numberofinstls_320L', 'numberofoutstandinstls_59L', 'numberofoverdueinstlmax_1039L', 'numberofoverdueinstlmaxdat_641D', 'numberofoverdueinstls_725L', 'outstandingamount_362A', 'overdueamount_659A', 'overdueamountmax2_14A', 'overdueamountmax2date_1142D', 'overdueamountmax_155A', 'overdueamountmaxdatemonth_365T', 'periodicityofpmts_837L', 'prolongationcount_599L', 'purposeofcred_426M', 'residualamount_856A', 'subjectrole_182M', 'totalamount_996A']
        self.active_contracts_columns.remove("financialinstitution_591M")
        self.active_contracts_columns.remove("classificationofcontr_13M")
        self.active_contracts_columns.remove("contractst_545M")
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

        # active contracts
        column = "dateofcredstart_739D"
        mask = table[column].is_not_null()
        assert(column in self.active_contracts_columns)
        dataset.set("credit_bureau_a_active_contracts_1", table.filter(mask)[self.active_contracts_columns + self.service_columns])

        # close contracts
        column = "dateofcredstart_181D"
        mask = table[column].is_not_null()
        assert(column in self.close_contracts_columns)
        dataset.set("credit_bureau_a_close_contracts_1", table.filter(mask)[self.close_contracts_columns + self.service_columns])

        # other
        dataset.set("credit_bureau_a_other_1", table[self.other_columns + self.service_columns])

        dataset.delete(self.table_name)
        return dataset, columns_info
