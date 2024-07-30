from unittest import TestCase
from src.functions.data_prep.misc_functions import remove_column_if_not_in_final_features

class Test(TestCase):
    def test_remove_column_if_not_in_final_features(self):
        final_features = ['SECHAB_dummie_G', 'SECPRODCOM_dummie_CREX', 'SECTYPEPROD_dummie_CP ', 'SEPRCFAC_dummie_X', 'SEPRCPLAN_dummie_X', 'SEBREPORT_dummie_2.0', 'SEBREPORT_dummie_4.0', 'SEBREPORT_dummie_6.0', 'SEBREPORT_dummie_7.0', 'SEBREPRCVT_dummie_nan', 'SECSITFAM_dummie_2.0', 'SECSITFAM_dummie_3.0', 'SECTEL_dummie_2.0', 'SECTEL_dummie_3.0', 'SEINACT_dummie_1.0', 'SEINACT_dummie_2.0', 'SEMSRECDIVERS_dummie_nan', 'SENBDOS_dummie_nan', 'SEPHASEA_dummie_4.0', 'SEPHNBORIG__2_dummie_nan', 'SEQUANT_dummie_5.0', 'SEQUANT_dummie_20.0', 'SECURRENTDPD_dummie_26.0', 'SECURRENTDPD_dummie_28.0', 'MM_EntryDate_LastPayment_dummie_-1.0', 'MM_EntryDate_LastPayment_dummie_1.0', 'MM_SEDDERSSR_EntryDate_dummie_1.0', 'MM_SEDDERSSR_EntryDate_dummie_2.0', 'MM_SEDDREG_EntryDate_dummie_nan', 'MM_SEDPHASEA_EntryDate_dummie_1.0', 'MM_SEDPHASEA_EntryDate_dummie_2.0', 'Criterion_Multiclass_dummie_1.0', 'EntryMonth_date_WeekDay_NB_dummie_2.0', 'EntryMonth_date_WeekDay_NB_dummie_6.0', 'Outstanding', 'SEANCPROF', 'SEBMONTSSR', 'SEBRGEAT', 'SEDECOUVERT', 'SEMCRD', 'SEMIR', 'SEMMENS', 'SEMMENSP', 'SEMREGORIG', 'SEMREPORTE', 'SEMSREC', 'SEMSRECAGIOS', 'SEMSRECCAP', 'SEPHNBORIG__7', 'SEPHNBORIG', 'SETODUCLT', 'SETODUDOSS', 'SEMAXDPDEVER', 'MM_SEDECH1_EntryDate', 'MM_SEDFIN_EntryDate', 'YY_SEDNAIS_EntryDate']
        numerical_cols = ['Outstanding', 'SEANCPROF', 'SEBMONTSSR', 'SEBRGEAT', 'SEDECOUVERT', 'SEMCRD', 'SEMIR', 'SEMMENS', 'SEMMENSP', 'SEMREGORIG', 'SEMREPORTE', 'SEMSREC', 'SEMSRECAGIOS', 'SEMSRECCAP', 'SEPHNBORIG__7', 'SEPHNBORIG', 'SETODUCLT', 'SETODUDOSS', 'SEMAXDPDEVER', 'MM_SEDECH1_EntryDate', 'MM_SEDFIN_EntryDate', 'YY_SEDNAIS_EntryDate']
        keep_cols = []
        empty_list = []

        final_features, numerical_cols = remove_column_if_not_in_final_features(final_features, numerical_cols, keep_cols)

        self.assertIsNot(numerical_cols, empty_list, msg = "remove_column_if_not_in_final_features() outputs empty numerical column")
        self.assertIsNotNone(numerical_cols, msg = "remove_column_if_not_in_final_features() outputs empty numerical column")

