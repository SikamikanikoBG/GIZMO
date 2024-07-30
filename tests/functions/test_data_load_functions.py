import unittest
import pandas as pd
from unittest.mock import patch
from src.functions.data_load_functions import check_separator_csv_file
from src.functions.printing_and_logging import print_and_log

class TestCheckSeparatorCSVFile(unittest.TestCase):
    @patch('pandas.read_csv')
    def test_check_separator_csv_file_with_valid_separator(self, mock_read_csv):
        # Arrange
        input_data_folder_name = 'C:/temp/gizmo/jizzmo/input_data'
        input_data_project_folder = '/project1'
        input_file = 'data.csv'
        mock_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_read_csv.return_value = mock_df

        # Act
        result = check_separator_csv_file(input_data_folder_name, input_data_project_folder, mock_df, input_file)

        # Assert
        self.assertTrue(result.equals(mock_df))
        mock_read_csv.assert_called_once_with(input_data_folder_name + input_data_project_folder + '/' + input_file, sep=',')

    @patch('pandas.read_csv')
    @patch('src.functions.data_load_functions.print_and_log')
    def test_check_separator_csv_file_with_invalid_separator(self, mock_print_and_log, mock_read_csv):
        # Arrange
        input_data_folder_name = 'C:/temp/gizmo/jizzmo/input_data'
        input_data_project_folder = '/project1'
        input_file = 'data.csv'
        mock_df = pd.DataFrame({'A': [1]})
        mock_read_csv.side_effect = [pd.errors.ParserError, mock_df]

        # Act
        with self.assertRaises(SystemExit):
            check_separator_csv_file(input_data_folder_name, input_data_project_folder, mock_df, input_file)

        # Assert
        self.assertEqual(mock_read_csv.call_count, 2)
        mock_read_csv.assert_any_call(input_data_folder_name + input_data_project_folder + '/' + input_file, sep=',')
        mock_read_csv.assert_any_call(input_data_folder_name + input_data_project_folder + '/' + input_file, sep=';')
        mock_print_and_log.assert_called_once_with('ERROR: input data separator not any of the following ,;', 'RED')

if __name__ == '__main__':
    unittest.main()
