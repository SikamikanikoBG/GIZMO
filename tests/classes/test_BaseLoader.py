import unittest
from unittest.mock import patch, MagicMock
from src.classes.BaseLoader import BaseLoader

class TestBaseLoader(unittest.TestCase):

    @patch('src.classes.BaseLoader.pd')
    @patch('src.classes.BaseLoader.print_and_log')
    def test_load_from_csv(self, mock_print_and_log, mock_pd):
        mock_pd.read_csv.return_value = MagicMock()
        input_data_folder_name = 'data_folder/'
        input_data_project_folder = 'project_folder/'
        file = 'test.csv'
        df = BaseLoader.load_from_csv(input_data_folder_name, input_data_project_folder, file)
        mock_pd.read_csv.assert_called_once_with(input_data_folder_name + input_data_project_folder + '/' + file)
        self.assertIsInstance(df, MagicMock)

    @patch('src.classes.BaseLoader.pd')
    @patch('src.classes.BaseLoader.print_and_log')
    def test_load_from_parquet(self, mock_print_and_log, mock_pd):
        mock_pd.read_parquet.return_value = MagicMock()
        input_data_folder_name = 'data_folder/'
        input_data_project_folder = 'project_folder/'
        file = 'test.parquet'
        df = BaseLoader.load_from_parquet(input_data_folder_name, input_data_project_folder, file)
        mock_pd.read_parquet.assert_called_once_with(input_data_folder_name + input_data_project_folder + '/' + file, engine='pyarrow')
        self.assertIsInstance(df, MagicMock)

    @patch('src.classes.BaseLoader.print_and_log')
    @patch('src.classes.BaseLoader.import_module')
    @patch('src.classes.BaseLoader.pd')
    def test_data_load_prep(self, mock_pd, mock_import_module, mock_print_and_log):
        mock_pd.read_csv.return_value = MagicMock()
        mock_pd.read_parquet.return_value = MagicMock()
        mock_import_module.run.return_value = MagicMock()
        mock_import_module.calculate_criterion.return_value = MagicMock()
        mock_pd.DataFrame.return_value = MagicMock()
        in_data_folder = 'data_folder/'
        in_data_proj_folder = 'project_folder/'
        params = {
            "main_table": "test.csv",
            "custom_calculations": "custom_calc",
            "additional_tables": ["additional1.csv", "additional2.parquet"],
            "under_sampling": True,
            "observation_date_column": "date_column",
            "periods_to_exclude": ["period1", "period2"],
            "criterion_column": "criterion_column"  # Add the missing key
        }
        predict_module = MagicMock()
        base_loader = BaseLoader(params, predict_module)
        base_loader.data_load_prep(in_data_folder, in_data_proj_folder)
        self.assertIsInstance(base_loader.in_df, MagicMock)
        self.assertIsInstance(base_loader.in_df_f, MagicMock)
        self.assertEqual(len(base_loader.additional_files_df_dict), 2)

    @patch('src.classes.BaseLoader.print_and_log')
    @patch('src.classes.BaseLoader.pq')
    @patch('src.classes.BaseLoader.pickle')
    def test_data_load_train(self, mock_pickle, mock_pq, mock_print_and_log):
        mock_pq.read_table.return_value = MagicMock()
        mock_pq.read_table.to_pandas.return_value = MagicMock()
        mock_pickle.load.return_value = MagicMock()
        output_data_folder_name = 'output_folder/'
        input_data_project_folder = 'project_folder/'
        params = {
            "under_sampling": True
        }
        predict_module = MagicMock()
        base_loader = BaseLoader(params, predict_module)
        base_loader.data_load_train(output_data_folder_name, input_data_project_folder)
        self.assertIsInstance(base_loader.in_df, MagicMock)
        self.assertIsInstance(base_loader.in_df_f, MagicMock)
        self.assertIsInstance(base_loader.final_features, MagicMock)

if __name__ == '__main__':
    unittest.main()
