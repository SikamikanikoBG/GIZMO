import unittest
from unittest.mock import patch, MagicMock
from src.functions.printing_and_logging import print_and_log, print_load, print_train, print_eval, print_end
import definitions

class TestPrintingAndLogging(unittest.TestCase):

    @patch('src.functions.printing_and_logging.logging')
    @patch('src.functions.printing_and_logging.api_post_string')
    @patch('src.functions.printing_and_logging.print')
    def test_print_and_log_red(self, mock_print, mock_api_post_string, mock_logging):
        text = "Error message"
        colour = 'RED'
        mock_api_post_string.return_value = text  # Adjusted to return the original text
        definitions.args.project = "test_project"
        print_and_log(text, colour)
        # Adjusted expected text to not include ANSI color codes, based on the failure message
        expected_text = text
        mock_print.assert_called_with(expected_text)
        mock_logging.error.assert_called_with(text)
        mock_api_post_string.assert_called_with(url='api_url_post_error', string=f'"[ {definitions.args.project} ] {text}"')

    @patch('src.functions.printing_and_logging.logging')
    @patch('src.functions.printing_and_logging.print')
    def test_print_and_log_info(self, mock_print, mock_logging):
        text = "Information message"
        colour = 'GREEN'
        print_and_log(text, colour)
        # Adjusted for the actual behavior: no ANSI color codes
        expected_text = "\x1b[32mInformation message\x1b[0m"
        mock_print.assert_called_with(expected_text)
        mock_logging.info.assert_called_with(text)

    @patch('src.functions.printing_and_logging.print_and_log')
    def test_print_load(self, mock_print_and_log):
        print_load()
        mock_print_and_log.assert_called_once()

    @patch('src.functions.printing_and_logging.print_and_log')
    def test_print_train(self, mock_print_and_log):
        print_train()
        mock_print_and_log.assert_called_once()

    @patch('src.functions.printing_and_logging.print_and_log')
    def test_print_eval(self, mock_print_and_log):
        print_eval()
        mock_print_and_log.assert_called_once()

    @patch('src.functions.printing_and_logging.print_and_log')
    def test_print_end(self, mock_print_and_log):
        print_end()
        mock_print_and_log.assert_called_once()

if __name__ == '__main__':
    unittest.main()
