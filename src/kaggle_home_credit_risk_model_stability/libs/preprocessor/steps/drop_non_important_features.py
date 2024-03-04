import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

    
class DropNonImportantFeaturesStep:
    def __init__(self):
        self.important_columns = ['dateofbirth_337D', 'days120_123L', 'days180_256L', 'days30_165L', 'days360_512L', 'days90_310L', 'description_5085714M', 'education_1103M', 'numberofqueries_373L', 'pmtaverage_3A', 'pmtaverage_4527227A', 'pmtscount_423L', 'pmtssum_45A', 'requesttype_4525192L', 'responsedate_4527233D', 'responsedate_4917613D', 'riskassesment_302T', 'thirdquarter_1082L', 'amtinstpaidbefduel24m_4187115A', 'annuity_780A', 'applicationcnt_361L', 'applicationscnt_867L', 'avgdpdtolclosure24_3658938P', 'avgmaxdpdlast9m_3716943P', 'bankacctype_710L', 'cardtype_51L', 'clientscnt12m_3712952L', 'clientscnt6m_3712949L', 'cntpmts24_3658933L', 'credtype_322L', 'currdebt_22A', 'datelastinstal40dpd_247D', 'deferredmnthsnum_166L', 'disbursedcredamount_1113A', 'disbursementtype_67L', 'eir_270L', 'homephncnt_628L', 'inittransactioncode_186L', 'interestrate_311L', 'isbidproduct_1095L', 'isdebitcard_729L', 'lastdelinqdate_224D', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastst_736L', 'mastercontrelectronic_519L', 'mastercontrexist_109L', 'maxdbddpdtollast12m_3658940P', 'maxdebt4_972A', 'maxdpdinstldate_3546855D', 'maxdpdlast24m_143P', 'maxdpdlast3m_392P', 'maxdpdlast6m_474P', 'maxdpdlast9m_1059P', 'maxdpdtolerance_374P', 'mobilephncnt_593L', 'numincomingpmts_3546848L', 'numinstlallpaidearly3d_817L', 'numinstlsallpaid_934L', 'numinstlswithdpd10_728L', 'numinstpaidearly3d_3546850L', 'numinsttopaygr_769L', 'numinstunpaidmax_3546851L', 'numinstunpaidmaxest_4493212L', 'pctinstlsallpaidlate1d_3546856L', 'pmtnum_254L', 'price_1097A', 'totaldebt_9A', 'twobodfilling_608L', 'validfrom_1069D', 'max_currdebt_94A', 'max_maxdpdtolerance_577P', 'max_outstandingdebt_522A', 'first_actualdpd_943P', 'mean_credacc_actualbalance_314A', 'mean_currdebt_94A', 'mean_maxdpdtolerance_577P', 'mean_outstandingdebt_522A', 'max_employedfrom_700D', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'min_education_1138M', 'first_education_1138M', 'last_education_1138M', 'last_rejectreason_755M', 'last_rejectreasonclient_4145042M', 'mode_education_1138M', 'mode_rejectreasonclient_4145042M', 'max_familystate_726L', 'max_pmtnum_8L', 'max_status_219L', 'max_tenor_203L', 'min_credtype_587L', 'min_familystate_726L', 'min_inittransactioncode_279L', 'min_isbidproduct_390L', 'first_familystate_726L', 'first_status_219L', 'last_credtype_587L', 'last_familystate_726L', 'min_num_group1', 'first_num_group1', 'max_amount_4527230A', 'mean_amount_4527230A', 'max_num_group1_3', 'min_num_group1_3', 'min_num_group1_4', 'min_num_group1_5', 'max_classificationofcontr_1114M', 'max_contractst_516M', 'max_contracttype_653M', 'max_periodicityofpmts_997M', 'max_pmtmethod_731M', 'max_purposeofcred_722M', 'max_subjectrole_326M', 'max_subjectrole_43M', 'min_classificationofcontr_1114M', 'min_contractst_516M', 'min_contracttype_653M', 'min_periodicityofpmts_997M', 'min_pmtmethod_731M', 'min_purposeofcred_722M', 'min_subjectrole_326M', 'min_subjectrole_43M', 'first_classificationofcontr_1114M', 'first_contractst_516M', 'first_contracttype_653M', 'first_periodicityofpmts_997M', 'first_purposeofcred_722M', 'first_subjectrole_326M', 'first_subjectrole_43M', 'last_classificationofcontr_1114M', 'last_contracttype_653M', 'last_periodicityofpmts_997M', 'last_pmtmethod_731M', 'last_purposeofcred_722M', 'last_subjectrole_326M', 'last_subjectrole_43M', 'mode_classificationofcontr_1114M', 'mode_purposeofcred_722M', 'mode_subjectrole_326M', 'mode_subjectrole_43M', 'max_birth_259D', 'min_birth_259D', 'first_birth_259D', 'last_birth_259D', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'min_education_927M', 'min_language1_981M', 'first_education_927M', 'first_language1_981M', 'last_education_927M', 'last_empladdr_district_926M', 'last_empladdr_zipcode_114M', 'last_language1_981M', 'max_contaddr_matchlist_1032L', 'max_empl_employedtotal_800L', 'max_familystate_447L', 'max_housingtype_772L', 'max_incometype_1044T', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_sex_738L', 'min_contaddr_matchlist_1032L', 'min_empl_employedtotal_800L', 'min_familystate_447L', 'min_housetype_905L', 'min_housingtype_772L', 'min_incometype_1044T', 'min_maritalst_703L', 'min_personindex_1023L', 'min_persontype_1072L', 'min_persontype_792L', 'min_relationshiptoclient_415T', 'min_relationshiptoclient_642T', 'min_role_993L', 'min_sex_738L', 'first_contaddr_matchlist_1032L', 'first_contaddr_smempladdr_334L', 'first_empl_employedtotal_800L', 'first_familystate_447L', 'first_housetype_905L', 'first_incometype_1044T', 'first_maritalst_703L', 'first_personindex_1023L', 'first_persontype_1072L', 'first_persontype_792L', 'first_role_993L', 'first_safeguarantyflag_411L', 'first_sex_738L', 'last_contaddr_matchlist_1032L', 'last_empl_industry_691L', 'last_familystate_447L', 'last_housetype_905L', 'last_housingtype_772L', 'last_incometype_1044T', 'last_maritalst_703L', 'last_sex_738L', 'min_num_group1_8', 'first_num_group1_8', 'min_num_group1_9', 'min_num_group1_10']
        
    def process_train_dataset(self, df, columns_info):  
        return self.process(df, columns_info)
        
    def process_test_dataset(self, df, columns_info):
        return self.process(df, columns_info)
    
    def process(self, df, columns_info):
        for column in df.columns:
            if (column in ["target", "WEEK_NUM", "case_id"]):
                continue
            if (column not in self.important_columns):
                df = df.drop(column)
        return df, columns_info