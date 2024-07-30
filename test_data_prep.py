"""
    This module is the main entry point for the application. It handles command-line arguments, loads and runs the
    appropriate modules based on the arguments, and logs the execution time.

    Attributes:
        parser (argparse.ArgumentParser): An instance of the ArgumentParser class to handle command-line arguments.

        args (argparse.Namespace): The parsed command-line arguments.

        help_need (str): The value of the '--h' argument, indicating whether the user needs help.

        module (object): An instance of the ModuleClass, which is determined based on the command-line arguments.
"""
import argparse
import sys
import warnings
from importlib import import_module
import definitions

from src.functions.printing_and_logging import print_and_log

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # -- session start --

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--tag', type=str, help='Tag the training_flows session. Optional')
    parser.add_argument('--project', type=str,
                        help='name of the project. Should be the same as the input folder and the param file.')
    parser.add_argument('--session', type=str, help='Train session folder to generate evaluation docx')
    parser.add_argument('--model', type=str, help='Model to evaluate. xgb, rf, dt, lr')
    parser.add_argument('--data_prep_module', type=str, help='Data prep module to run')
    parser.add_argument('--train_module', type=str, help='Training module to run')
    parser.add_argument('--predict_module', type=str, help='Training module to run') # ardi
    parser.add_argument('--pred_data_prep', type=str, help='Which data prep to be used for predict') # ardi
    parser.add_argument('--eval_module', type=str, help='Which data prep to be used for predict')
    parser.add_argument('--nb_tree_features', type=int, help='Nb of max features for the tree models')
    parser.add_argument('--main_model', type=str, help='Main model to predict')
    parser.add_argument('--h', type=str, help='You need help...')

    args = parser.parse_args()
    args.project = "bg_stage2"
    args.data_prep_module = "standard"

    definitions.args = args

    # todo: check if ram check is needed
    #if psutil.virtual_memory()[4] / 1e9 < 5:
    #    print_and_log('[ RAM MEMORY CHECK ] Free RAM memory < 50GB. Program aborted...', '')
    #    sys.exit()


    module = args.data_prep_module.lower()
    module_lib = import_module(f'src.flows.data_prep_flows.{module}')

    module = module_lib.ModuleClass(args=args)
    module.prepare()
    
    try:
        module.run()
    except Exception as e:
        print_and_log(e, "RED")
    module.run_time_calc()
    print_and_log(module.run_time, "")
