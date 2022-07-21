import argparse
import sys
import warnings
from importlib import import_module

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
    parser.add_argument('--h', type=str, help='You need help...')
    args = parser.parse_args()

    help_need = args.h
    if help_need:
        print_and_log(parser.parse_known_args(), "RED")
        sys.exit()

    if args.data_prep_module:
        data_prep_module = args.data_prep_module.lower()
        module_lib = import_module(f'src.data_prep_flows.{data_prep_module}')
        data_prep = module_lib.ModuleClass(args=args)
        data_prep.run()
        data_prep.run_time_calc()
        print_and_log(data_prep.run_time, "")
    elif args.train_module:
        train_module_arg = args.train_module.lower()
        module_lib = import_module(f'src.training_flows.{train_module_arg}')
        train_module = module_lib.ModuleClass(args=args)
        train_module.run()
        train_module.run_time_calc()
        print_and_log(train_module.run_time, "")
