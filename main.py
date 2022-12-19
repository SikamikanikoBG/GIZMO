import argparse
import sys
import warnings
from importlib import import_module
import psutil
import definitions

from src.functions.printing_and_logging import print_and_log

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # -- session start --


    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Required positional argument
    #parser.add_argument('--run', type=str, help='After run write which command to execute: load, train, eval')
    parser.add_argument('--tag', type=str, help='Tag the training_flows session. Optional')
    parser.add_argument('--project', type=str,
                        help='name of the project. Should  the same as the input folder and the param file.')
    parser.add_argument('--session', type=str, help='Train session folder to generate evaluation docx')
    parser.add_argument('--model', type=str, help='Model to evaluate. xgb, rf, dt, lr')

    parser.add_argument('--data_prep_module', type=str, help='Data prep module to run')
    parser.add_argument('--train_module', type=str, help='Training module to run')
    parser.add_argument('--predict_module', type=str, help='Training module to run')
    parser.add_argument('--pred_data_prep', type=str, help='Which data prep to be used for predict')

    parser.add_argument('--tp', type=str, help='Which data prep to be used for predict')
    parser.add_argument('--sl', type=str, help='Which data prep to be used for predict')
    parser.add_argument('--period', type=str, help='Which data prep to be used for predict')
    parser.add_argument('--nb_tree_features', type=int, help='Nb of max features for the tree models')

    parser.add_argument('--main_model', type=str, help='Main model to predict')

    parser.add_argument('--h', type=str, help='You need help...')
    args = parser.parse_args()
    definitions.args = args

    help_need = args.h
    if help_need:
        print_and_log(parser.parse_known_args(), "RED")
        sys.exit()

    # todo: check if ram check is needed
    #if psutil.virtual_memory()[4] / 1e9 < 5:
    #    print_and_log('[ RAM MEMORY CHECK ] Free RAM memory < 50GB. Program aborted...', '')
    #    sys.exit()

    if args.data_prep_module:
        module = args.data_prep_module.lower()
        module_lib = import_module(f'src.flows.data_prep_flows.{module}')
    elif args.train_module:
        module = args.train_module.lower()
        module_lib = import_module(f'src.flows.training_flows.{module}')
    elif args.predict_module:
        module = args.predict_module.lower()
        module_lib = import_module(f'src.flows.predict_flows.{module}')

    module = module_lib.ModuleClass(args=args)
    module.prepare()
    module.start_logging()
    try:
        module.run()
    except Exception as e:
        print_and_log(e, "RED")
    module.run_time_calc()
    print_and_log(module.run_time, "")
