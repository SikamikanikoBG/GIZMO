import json
import logging
import os
from datetime import datetime
from colorama import Fore, Style
import definitions
from src import BaseLoader, print_and_log

class SessionManager:
    def __init__(self, args):
        """
        Initialize the SessionManager object with the provided arguments.

        Args:
            args: Arguments for the session.

        Attributes:
            check4_time_runtime: Runtime for check 4.

            check3_time_runtime: Runtime for check 3.

            check2_time_runtime: Runtime for check 2.

            check1_time_runtime: Runtime for check 1.

            start_time: Start time of the session.

            end_time: End time of the session.

            run_time: Total runtime of the session.

            data_load_time: Time taken for data loading.

            check1_time: Time for check 1.

            check2_time: Time for check 2.

            check3_time: Time for check 3.

            check4_time: Time for check 4.

            session_id: Session ID.

            params: Parameters.

            args: Arguments for the session.

            session_to_eval: Session to evaluate.

            model_arg: Model argument.

            project_name: Project name.

            tag: Tag for the session.

            log_file_name: Log file name.

            log_folder_name: Log folder name.

            session_folder_name: Session folder name.

            session_id_folder: Session ID folder.

            input_data_folder_name: Input data folder name.

            input_data_project_folder: Input data project folder.

            output_data_folder_name: Output data folder name.

            functions_folder_name: Functions folder name.

            params_folder_name: Params folder name.

            implemented_folder: Implemented models folder.
        """
        self.check4_time_runtime = None
        self.check3_time_runtime = None
        self.check2_time_runtime = None
        self.check1_time_runtime = None
        self.start_time = datetime.now()
        self.end_time = None
        self.run_time = None
        self.data_load_time = None
        self.check1_time = None
        self.check2_time = None
        self.check3_time = None
        self.check4_time = None
        self.session_id = None
        self.params = None
        self.args = args
        self.session_to_eval = self.args.session
        self.model_arg = self.args.model
        self.project_name = self.args.project
        self.tag = self.args.tag if self.args.tag else 'no_tag'

        # Create timestamp without colons and spaces
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        # Generate log filename based on module type
        if self.args.data_prep_module:
            module_prefix = f'load_{self.args.data_prep_module}'
        elif self.args.train_module:
            module_prefix = f'train_{self.args.train_module}'
        elif self.args.predict_module:
            module_prefix = f'predict_{self.args.predict_module}'
        elif self.args.eval_module:
            module_prefix = f'eval_{self.args.eval_module}'
        else:
            module_prefix = 'unknown'

        self.log_file_name = f"{module_prefix}_{self.project_name}_{timestamp}_{self.tag}.log"

        # Set up folder structure using os.path.join
        self.log_folder_name = os.path.join(definitions.EXTERNAL_DIR, 'logs')
        self.session_folder_name = os.path.join(definitions.EXTERNAL_DIR, 'sessions')
        os.makedirs(self.session_folder_name, exist_ok=True)
        self.session_id_folder = None
        self.input_data_folder_name = os.path.join(definitions.ROOT_DIR, 'input_data')
        self.input_data_project_folder = self.project_name
        self.output_data_folder_name = os.path.join(definitions.ROOT_DIR, 'output_data')
        self.functions_folder_name = os.path.join(definitions.ROOT_DIR, 'src')
        self.params_folder_name = os.path.join(definitions.EXTERNAL_DIR, 'params')
        self.implemented_folder = os.path.join(definitions.EXTERNAL_DIR, 'implemented_models')
        
        # Create basic folder structure
        self.create_base_folders()
        self.start_logging()

        # Import parameters
        try:
            params_file = os.path.join(self.params_folder_name, f'params_{self.project_name}.json')
            with open(params_file) as json_file:
                self.params = json.load(json_file)
                definitions.params = self.params
            
            # Load parameter values
            self.criterion_column = self.params['criterion_column']
            self.missing_treatment = self.params["missing_treatment"]
            self.observation_date_column = self.params["observation_date_column"]
            self.columns_to_exclude = self.params["columns_to_exclude"]
            self.periods_to_exclude = self.params["periods_to_exclude"]
            self.t1df_period = self.params["t1df"]
            self.t2df_period = self.params["t2df"]
            self.t3df_period = self.params["t3df"]
            self.lr_features = self.params["lr_features"]
            self.cut_offs = self.params["cut_offs"]
            self.under_sampling = self.params['under_sampling']
            self.optimal_binning_columns = self.params['optimal_binning_columns']
            self.main_table = self.params["main_table"]
            self.columns_to_include = self.params["columns_to_include"]

        except Exception as e:
            print(Fore.RED + 'ERROR: params file not available' + Style.RESET_ALL)
            print(e)
            logging.error(e)
            quit()

        # Initialize loader
        self.loader = BaseLoader(params=self.params, predict_module=self.args.predict_module)
        if self.args.data_prep_module:
            self.loader.data_load_prep(in_data_folder=self.input_data_folder_name,
                                     in_data_proj_folder=self.input_data_project_folder)
        elif self.args.train_module:
            self.loader.data_load_train(output_data_folder_name=self.output_data_folder_name,
                                        input_data_project_folder=self.input_data_project_folder)
        elif self.args.predict_module:
            pass

    def prepare(self):
        """
        Orchestrates the preparation of the session run by creating necessary folders
        and initializing the appropriate session type.
        """
        # First create base folders
        self.create_base_folders()
        
        # Then create specific session folder based on module type
        if self.args.train_module:
            self.create_train_session_folder()
        elif self.args.eval_module:
            self.create_eval_session_folder()
        elif self.args.predict_module:
            self.create_predict_session_folder()
        elif self.args.data_prep_module:
            self.create_data_prep_session_folder()

    def create_base_folders(self):
        """
        Creates the base folder structure required for the application.
        This includes directories for logs, sessions, input/output data, etc.
        """
        base_folders = [
            definitions.EXTERNAL_DIR,
            self.log_folder_name,
            self.session_folder_name,
            self.input_data_folder_name,
            self.output_data_folder_name,
            self.functions_folder_name,
            self.params_folder_name,
            self.implemented_folder
        ]
        
        # Create base folders
        for folder in base_folders:
            try:
                os.makedirs(folder, exist_ok=True)
                print_and_log(f"[ FOLDERS ] Verified directory: {folder}", "")
            except Exception as e:
                print_and_log(f"[ ERROR ] Failed to create directory {folder}: {str(e)}", "RED")
                raise

        # Create project-specific output directory
        try:
            project_output_dir = os.path.join(self.output_data_folder_name, self.input_data_project_folder)
            os.makedirs(project_output_dir, exist_ok=True)
            print_and_log(f"[ FOLDERS ] Created project output directory: {project_output_dir}", "")
        except Exception as e:
            print_and_log(f"[ ERROR ] Failed to create project output directory: {str(e)}", "RED")
            raise

    def create_train_session_folder(self):
        """
        Creates a training session folder with proper naming and structure.
        """
        print_and_log('[ SESSION ] Creating training session folder', 'YELLOW')
        
        # Format timestamp without spaces
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Generate session ID
        self.session_id = f'TRAIN_{self.project_name}_{timestamp}_{self.tag}'
        
        # Create full session path with proper separator
        self.session_id_folder = os.path.join(self.session_folder_name, self.session_id)
        
        try:
            # Create main session directory
            os.makedirs(self.session_id_folder, exist_ok=True)
            
            # Create model subdirectories
            for model_dir in ['dt', 'rf', 'xgb', 'lr']:
                model_path = os.path.join(self.session_id_folder, model_dir)
                plots_path = os.path.join(model_path, 'plots')
                models_path = os.path.join(model_path, 'models')
                metrics_path = os.path.join(model_path, 'metrics')
                
                for path in [model_path, plots_path, models_path, metrics_path]:
                    os.makedirs(path, exist_ok=True)
                
            # Create data directory
            data_dir = os.path.join(self.session_id_folder, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            print_and_log(f'[ SESSION ] Created folder {self.session_id_folder}', '')
            
        except Exception as e:
            print_and_log(f'[ ERROR ] Failed to create training session folder: {str(e)}', 'RED')
            raise

    def create_eval_session_folder(self):
        """
        Creates an evaluation session folder with proper naming and structure.
        """
        print_and_log('[ SESSION ] Creating evaluation session folder', 'YELLOW')
        
        # Format timestamp without spaces
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Generate session ID
        self.session_id = f'EVAL_{self.project_name}_{timestamp}_{self.tag}'
        
        # Create full session path with proper separator
        self.session_id_folder = os.path.join(self.session_folder_name, self.session_id)
        
        try:
            # Create evaluation directory structure
            os.makedirs(self.session_id_folder, exist_ok=True)
            
            # Create standard subdirectories
            subdirs = ['data', 'results', 'metrics']
            for subdir in subdirs:
                os.makedirs(os.path.join(self.session_id_folder, subdir), exist_ok=True)
                
            print_and_log(f'[ SESSION ] Created folder {self.session_id_folder}', '')
            
        except Exception as e:
            print_and_log(f'[ ERROR ] Failed to create evaluation folder: {str(e)}', 'RED')
            raise

    def create_predict_session_folder(self):
        """
        Creates a prediction session folder with proper naming and structure.
        """
        print_and_log('[ SESSION ] Creating prediction session folder', 'YELLOW')
        
        # Format timestamp without spaces
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Generate session ID
        self.session_id = f'PREDICT_{self.project_name}_{timestamp}_{self.tag}'
        
        # Create full session path with proper separator
        self.session_id_folder = os.path.join(self.session_folder_name, self.session_id)
        
        try:
            # Create directory structure
            os.makedirs(self.session_id_folder, exist_ok=True)
            
            # Create subdirectories
            subdirs = ['predictions', 'metrics', 'models']
            for subdir in subdirs:
                os.makedirs(os.path.join(self.session_id_folder, subdir), exist_ok=True)
                
            print_and_log(f'[ SESSION ] Created folder {self.session_id_folder}', '')
            
        except Exception as e:
            print_and_log(f'[ ERROR ] Failed to create prediction folder: {str(e)}', 'RED')
            raise

    def create_data_prep_session_folder(self):
        """
        Creates a data preparation session folder with proper naming and structure.
        """
        print_and_log('[ SESSION ] Creating data prep session folder', 'YELLOW')
        
        # Format timestamp without spaces
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Generate session ID
        self.session_id = f'PREP_{self.project_name}_{timestamp}_{self.tag}'
        
        # Create full session path with proper separator
        self.session_id_folder = os.path.join(self.session_folder_name, self.session_id)
        
        try:
            # Create directory structure
            os.makedirs(self.session_id_folder, exist_ok=True)
            
            # Create subdirectories
            subdirs = ['raw', 'processed', 'features']
            for subdir in subdirs:
                os.makedirs(os.path.join(self.session_id_folder, subdir), exist_ok=True)
                
            print_and_log(f'[ SESSION ] Created folder {self.session_id_folder}', '')
            
        except Exception as e:
            print_and_log(f'[ ERROR ] Failed to create data prep folder: {str(e)}', 'RED')
            raise


    def start_logging(self):
        """Starts the logging process for the session"""
        log_file_path = os.path.join(self.log_folder_name, self.log_file_name)
        
        # Create or clear the log file
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        open(log_file_path, 'w').close()

        # Configure logging
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        print_and_log('[ LOGGING ] Start logging', 'YELLOW')

    def run_time_calc(self):
        """Calculates session runtime and formats output"""
        self.end_time = datetime.now()
        self.run_time = round(float((self.end_time - self.start_time).total_seconds()), 2)

        def format_time(seconds):
            if seconds > 60:
                return f"{seconds / 60:.2f} minutes"
            else:
                return f"{seconds}s"

        check_times = [self.check1_time, self.check2_time, self.check3_time, self.check4_time]
        check_time_runtimes = []
        for i, check_time in enumerate(check_times):
            if check_time:
                runtime = round(float((check_time - (check_times[i - 1] if i > 0 else self.start_time)).total_seconds()), 2)
                check_time_runtimes.append(format_time(runtime))

        check_time_log_str = ",\n".join(f"Check{i+1} time: {time}" for i, time in enumerate(check_time_runtimes))

        print_and_log(f"RUN time: {format_time(self.run_time)}, {check_time_log_str}", "YELLOW")
        if self.args.train_module:
            print_and_log(
                f"Session complete.\nRun standard eval session with: python main.py --project {self.project_name} "
                f"--eval_module standard --session \"{self.session_id}\"",
                'GREEN'
            )