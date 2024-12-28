"""
Main entry point for the application with integrated system resource management.
"""
import argparse
import sys
import warnings
from importlib import import_module
import psutil
import os
import platform
from typing import Tuple
import definitions
from src.functions.printing_and_logging import print_and_log

warnings.filterwarnings("ignore")

class SystemGuard:
    """Monitors and manages system resources for the application."""
    
    def __init__(self):
        self.system = platform.system()
        self.memory = psutil.virtual_memory()
        self.cpu_count = psutil.cpu_count()
        self.disk = psutil.disk_usage('/')
        
    def check_system_resources(self) -> Tuple[bool, str]:
        """
        Check if system meets minimum requirements.
        Returns:
            Tuple[bool, str]: (meets_requirements, message)
        """
        try:
            # Check memory
            total_ram_gb = self.memory.total / (1024 ** 3)
            available_ram_gb = self.memory.available / (1024 ** 3)
            
            # Check disk space
            free_disk_gb = self.disk.free / (1024 ** 3)
            
            # Check page file (Windows)
            if self.system == 'Windows':
                try:
                    import win32api
                    import win32file
                    total_pagefile = win32api.GlobalMemoryStatusEx()['TotalPageFile'] / (1024 ** 3)
                    if total_pagefile < 8:
                        return False, (
                            "WARNING: Windows page file is too small. Please increase it:\n"
                            "1. Open System Properties > Advanced > Performance Settings\n"
                            "2. Advanced > Virtual Memory > Change\n"
                            "3. Set custom size: Initial=8GB, Maximum=16GB or more"
                        )
                except ImportError:
                    pass  # Skip if win32api not available
            
            # Define minimum requirements
            MIN_RAM = 8  # GB
            MIN_FREE_RAM = 4  # GB
            MIN_DISK = 20  # GB
            MIN_CPU = 2  # cores
            
            if total_ram_gb < MIN_RAM:
                return False, f"System has only {total_ram_gb:.1f}GB RAM. Minimum {MIN_RAM}GB required."
                
            if available_ram_gb < MIN_FREE_RAM:
                return False, f"Only {available_ram_gb:.1f}GB RAM available. Minimum {MIN_FREE_RAM}GB required."
                
            if free_disk_gb < MIN_DISK:
                return False, f"Only {free_disk_gb:.1f}GB disk space available. Minimum {MIN_DISK}GB required."
                
            if self.cpu_count < MIN_CPU:
                return False, f"Only {self.cpu_count} CPU cores available. Minimum {MIN_CPU} required."
            
            return True, "System resources OK"
            
        except Exception as e:
            return False, f"Error checking system resources: {str(e)}"
    
    def optimize_resources(self):
        """Optimize system resources for the application."""
        try:
            # Calculate optimal thread count (leave 1 core free for system)
            optimal_threads = max(1, self.cpu_count - 1)
            
            # Calculate optimal memory limit (75% of available memory)
            optimal_memory_mb = int((self.memory.available * 0.75) / (1024 * 1024))
            
            # Set environment variables
            os.environ['NUMEXPR_MAX_THREADS'] = str(optimal_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['PYTHONMEM'] = str(optimal_memory_mb)
            
            # If on Windows, configure process priority
            if self.system == 'Windows':
                try:
                    import win32api
                    import win32process
                    win32process.SetPriorityClass(win32api.GetCurrentProcess(), 
                                                win32process.ABOVE_NORMAL_PRIORITY_CLASS)
                except ImportError:
                    pass
                    
            # # Configure numpy to use multiple threads but not all
            # try:
            #     import numpy as np
            #     np.set_num_threads(optimal_threads)
            # except ImportError:
            #     pass
                
            return True, f"Resources optimized: {optimal_threads} threads, {optimal_memory_mb}MB memory limit"
            
        except Exception as e:
            return False, f"Error optimizing resources: {str(e)}"
    
    def monitor_resources(self):
        """
        Returns current resource usage as a formatted string.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            return (f"Resource Usage:\n"
                   f"CPU: {cpu_percent}%\n"
                   f"Memory: {memory_percent}%\n"
                   f"Disk: {disk_percent}%")
        except:
            return "Resource monitoring unavailable"


def initialize_system():
    """Initialize system and check resources."""
    guard = SystemGuard()
    
    # Check system resources
    meets_requirements, message = guard.check_system_resources()
    print_and_log(message, "RED" if not meets_requirements else "")
    if not meets_requirements:
        return None
        
    # Optimize resources
    success, message = guard.optimize_resources()
    print_and_log(message, "RED" if not success else "")
    if not success:
        return None
        
    return guard


def main():
    """Main application entry point with system resource management."""
    # Initialize system guard
    guard = initialize_system()
    if guard is None:
        print_and_log("System requirements not met. Please check above messages.", "RED")
        sys.exit(1)
        
    # Start resource monitoring
    print_and_log(guard.monitor_resources(), "RED")

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Add arguments
    parser.add_argument('--tag', type=str, help='Tag the session. Optional')
    parser.add_argument('--project', type=str,
                       help='name of the project. Should be the same as the input folder and the param file.')
    parser.add_argument('--session', type=str, help='Train session folder to generate evaluation docx')
    parser.add_argument('--model', type=str, help='Model to evaluate. xgb, rf, dt, lr')
    parser.add_argument('--data_prep_module', type=str, help='Data prep module to run')
    parser.add_argument('--train_module', type=str, help='Training module to run')
    parser.add_argument('--predict_module', type=str, help='Training module to run')
    parser.add_argument('--pred_data_prep', type=str, help='Which data prep to be used for predict')
    parser.add_argument('--eval_module', type=str, help='Which data prep to be used for predict')
    parser.add_argument('--nb_tree_features', type=int, help='Nb of max features for the tree models')
    parser.add_argument('--main_model', type=str, help='Main model to predict')
    parser.add_argument('--h', type=str, help='You need help...')
    
    args = parser.parse_args()
    definitions.args = args

    if args.h:
        print_and_log(parser.parse_known_args(), "RED")
        sys.exit()

    # Module selection and execution
    try:
        # Determine module type and path
        module_config = {
            'data_prep_module': ('data_prep_flows', 'PREP'),
            'train_module': ('training_flows', 'TRAIN'),
            'predict_module': ('predict_flows', 'PREDICT'),
            'eval_module': ('eval_flows', 'EVAL')
        }
        
        module_type = None
        module_path = None
        session_type = None
        
        for arg_name, (path_suffix, session_prefix) in module_config.items():
            if getattr(args, arg_name):
                module = getattr(args, arg_name).lower()
                module_type = arg_name
                module_path = f'src.flows.{path_suffix}.{module}'
                session_type = session_prefix
                break
        
        if not module_path:
            print_and_log("No valid module specified", "RED")
            sys.exit(1)

        print_and_log(f"[ MODULE ] Initializing {session_type} module", "")
        
        # Import and initialize module
        module_lib = import_module(module_path)
        module = module_lib.ModuleClass(args=args)
        
        try:
            # Initialize session environment
            module.prepare()
            
            # Print initial resource state
            print_and_log("\nInitial " + guard.monitor_resources(), "YELLOW")
            
            # Run the module
            module.run()
            
            # Print final resource state
            print_and_log("\nFinal " + guard.monitor_resources(), "GREEN")
            
        except Exception as e:
            print_and_log(str(e), "RED")
            print_and_log("\nError state " + guard.monitor_resources(), "RED")
            raise
        
        module.run_time_calc()
        print_and_log(module.run_time, "")
        
    except Exception as e:
        print_and_log(f"Error during execution: {str(e)}", "RED")
        print_and_log("\nFinal error state:\n" + guard.monitor_resources(), "RED")
        sys.exit(1)


if __name__ == '__main__':
    main()