import os
import json
import logging
import hashlib
import threading
import subprocess
import gradio as gr
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from functools import wraps
import definitions
from queue import Queue, Empty
from typing import Optional, Tuple

import sys
import logging
from queue import Queue, Empty
from collections import deque
from typing import Optional, Tuple

class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.terminal = sys.stdout
        self.reset_logs()
        self.log = open(filename, "w")
        self.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

    def reset_logs(self):
        """Clear the log file"""
        with open(self.filename, 'w') as file:
            file.truncate(0)

    def read_logs(self):
        """Read and return the most recent logs"""
        sys.stdout.flush()
        with open(self.filename, "r") as f:
            log_content = f.readlines()
        
        # Get the latest 100 lines
        recent_lines = log_content[-100:]
        return ''.join(recent_lines)

def enqueue_output(out, queue):
    """Add output lines to queue"""
    for line in iter(out.readline, ''):  # Changed from b'' to '' since we're using text mode
        queue.put(line)
    out.close()

class ProcessExecutor:
    def __init__(self, logger):
        self.logger = logger
        self.sys_logger = Logger("output.log")
        sys.stdout = self.sys_logger  # Redirect stdout to our logger
        
    def run_pipeline_step(self, cmd: str, cancel_event: threading.Event, callback=None) -> Tuple[str, int]:
        """
        Run pipeline step with proper process management and real-time output handling
        """
        try:
            self.logger.info(f"Running command: {cmd}")
            print(f"Running command: {cmd}")  # This will go to both terminal and log file
            
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_queue = Queue()
            stderr_queue = Queue()
            
            stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
            stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            while True:
                if cancel_event.is_set():
                    process.terminate()
                    print("Operation cancelled")
                    return "Operation cancelled", -1
                
                return_code = process.poll()
                
                # Handle stdout
                try:
                    while True:  # Read all available output
                        line = stdout_queue.get_nowait()
                        line = line.strip()
                        if line:
                            print(line)  # This goes to log file
                            if callback:
                                callback(self.sys_logger.read_logs())
                except Empty:
                    pass
                
                # Handle stderr
                try:
                    while True:  # Read all available output
                        line = stderr_queue.get_nowait()
                        line = line.strip()
                        if line:
                            print(f"ERROR: {line}")  # This goes to log file
                            if callback:
                                callback(self.sys_logger.read_logs())
                except Empty:
                    pass
                
                if return_code is not None:
                    break
                    
                # Small sleep to prevent CPU spinning
                import time
                time.sleep(0.1)
            
            # Read final output
            output = self.sys_logger.read_logs()
            
            if return_code != 0:
                self.logger.error(f"Pipeline step failed: {output}")
                return f"Error: {output}", return_code
                
            return output, return_code
            
        except Exception as e:
            error_msg = f"Error running pipeline step: {str(e)}"
            self.logger.error(error_msg)
            print(error_msg)  # This goes to log file
            return error_msg, -1
@dataclass
class ExecutionStatus:
    is_running: bool
    message: str
    progress: float
    can_cancel: bool

class GizmoUI:
    def __init__(self):
        self.root_dir = definitions.ROOT_DIR
        self.external_dir = definitions.EXTERNAL_DIR
        self.project_structure = {
            'logs': os.path.join(self.root_dir, 'logs'),
            'input_data': os.path.join(self.root_dir, 'input_data'),
            'output_data': os.path.join(self.root_dir, 'output_data'),
            'params': os.path.join(self.root_dir, 'params'),
            'sessions': os.path.join(self.root_dir, 'sessions'),
            'implemented_models': os.path.join(self.root_dir, 'implemented_models')
        }
        for directory in self.project_structure.values():
            os.makedirs(directory, exist_ok=True)
        self.setup_logging()
        self.running_processes: Dict[str, threading.Event] = {}
        self.process_executor = ProcessExecutor(self.logger)
        self.load_config()
    
    def setup_logging(self):
        """Configure logging with proper formatting and rotation"""
        log_file = os.path.join(self.project_structure['logs'], 'gizmo_ui.log')
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger('GizmoUI')
        
    def load_config(self):
        """Load configuration settings"""
        config_path = os.path.join(self.root_dir, 'config.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {}
                self.logger.warning("No config file found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            self.config = {}

    def validate_project_name(self, project_name: str) -> bool:
        """Validate project name for security"""
        if not project_name or not isinstance(project_name, str):
            return False
        return all(c.isalnum() or c == '_' for c in project_name)

    def list_projects(self) -> List[str]:
        """List all available projects based on param files"""
        try:
            params_dir = self.project_structure['params']
            if not os.path.exists(params_dir):
                self.logger.error(f"Params directory not found at {params_dir}")
                return []
            param_files = []
            for f in os.listdir(params_dir):
                if f.startswith('params_') and f.endswith('.json'):
                    if not os.path.isdir(os.path.join(params_dir, f)):
                        param_files.append(f)
            projects = [f[7:-5] for f in param_files]
            self.logger.info(f"Found projects: {projects}")
            return sorted(projects)
        except Exception as e:
            self.logger.error(f"Error listing projects: {str(e)}")
            return []

    def load_param_form(self, project_name: str) -> Tuple[str, str, str, str, str, str, List, List, List]:
        """Load parameters into structured form format"""
        if not project_name:
            return "", "", "Missing", "", "", "", [], [], []
        
        try:
            param_path = os.path.join(self.project_structure['params'], f'params_{project_name}.json')
            with open(param_path, 'r') as f:
                params = json.load(f)
                
            # Convert secondary criteria to list format for DataFrame
            secondary = []
            for i, col in enumerate(params.get("secondary_criterion_columns", [])):
                secondary.append([str(i), str(col)])
            
            # Convert exclude columns to list format for DataFrame
            exclude_cols = [[col] for col in params.get("columns_to_exclude", [])] if params.get("columns_to_exclude") else []
            
            # Convert exclude periods to list format for DataFrame
            exclude_periods = [[period] for period in params.get("periods_to_exclude", [])] if params.get("periods_to_exclude") else []
            
            return (
                params.get("criterion_column", ""),
                params.get("observation_date_column", ""),
                params.get("missing_treatment", {}).get("Info", "Missing"),
                params.get("t1df", ""),
                params.get("t2df", ""),
                params.get("t3df", ""),
                secondary,
                exclude_cols,
                exclude_periods
            )
        except Exception as e:
            self.logger.error(f"Error loading parameters form: {str(e)}")
            return "", "", "Missing", "", "", "", [], [], []

    def save_param_form(self, 
                   project_name: str,
                   criterion: str,
                   obs_date: str,
                   missing: str,
                   t1: str,
                   t2: str,
                   t3: str,
                   secondary: pd.DataFrame,
                   excl_cols: pd.DataFrame,
                   excl_periods: pd.DataFrame) -> str:
        """Save parameters from structured form while preserving other fields"""
        try:
            # First load existing params to preserve other fields
            param_path = os.path.join(self.project_structure['params'], f'params_{project_name}.json')
            if os.path.exists(param_path):
                with open(param_path, 'r') as f:
                    existing_params = json.load(f)
            else:
                existing_params = {}

            # Process secondary criteria (convert DataFrame to list)
            if not secondary.empty:
                secondary_cols = secondary.iloc[:, 1].tolist()  # Take second column
                secondary_cols = [col for col in secondary_cols if pd.notna(col)]
            else:
                secondary_cols = existing_params.get("secondary_criterion_columns", [])

            # Process exclude columns (convert DataFrame to list)
            if not excl_cols.empty:
                exclude_columns = excl_cols.iloc[:, 0].tolist()  # Take first column
                exclude_columns = [col for col in exclude_columns if pd.notna(col)]
            else:
                exclude_columns = existing_params.get("columns_to_exclude", [])

            # Process exclude periods (convert DataFrame to list)
            if not excl_periods.empty:
                exclude_periods = excl_periods.iloc[:, 0].tolist()  # Take first column
                exclude_periods = [period for period in exclude_periods if pd.notna(period)]
            else:
                exclude_periods = existing_params.get("periods_to_exclude", [])

            # Update with new values while preserving existing structure
            params = {
                **existing_params,  # Preserve all existing fields
                "criterion_column": criterion if criterion else existing_params.get("criterion_column", ""),
                "observation_date_column": obs_date if obs_date else existing_params.get("observation_date_column", ""),
                "missing_treatment": {"Info": missing} if missing else existing_params.get("missing_treatment", {"Info": "Missing"}),
                "t1df": t1 if t1 else existing_params.get("t1df", ""),
                "t2df": t2 if t2 else existing_params.get("t2df", ""),
                "t3df": t3 if t3 else existing_params.get("t3df", ""),
                "secondary_criterion_columns": secondary_cols,
                "columns_to_exclude": exclude_columns,
                "periods_to_exclude": exclude_periods
            }
            
            # Backup existing file
            if os.path.exists(param_path):
                backup_path = param_path + f'.bak.{int(datetime.now().timestamp())}'
                os.rename(param_path, backup_path)
            
            # Save updated params
            with open(param_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            self.logger.info(f"Parameters saved for project {project_name}")
            return "Parameters saved successfully"
        except Exception as e:
            error_msg = f"Error saving parameters: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def run_pipeline_step(self, cmd: str, cancel_event: threading.Event) -> str:
        """Run pipeline step with proper process management"""
        try:
            print(cmd)
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            while True:
                if cancel_event.is_set():
                    process.terminate()
                    return "Operation cancelled"
                try:
                    process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                self.logger.error(f"Pipeline step failed: {stderr}")
                return f"Error: {stderr}"
            return stdout
        except Exception as e:
            self.logger.error(f"Error running pipeline step: {str(e)}")
            return f"Error: {str(e)}"

    def run_data_prep(self, project_name: str, tag: str) -> str:
        """Run data preparation module with sanitized tag"""
        if not project_name:
            return "Please select a project"

        safe_tag = tag.replace(':', '').replace(' ', '_')
        cmd = f"python main.py --project {project_name} --data_prep_module standard --tag {safe_tag}"

        current_dir = os.getcwd()
        try:
            os.chdir(self.root_dir)
            self.logger.info(f"Running data prep command: {cmd}")
            cancel_event = threading.Event()
            self.running_processes[project_name] = cancel_event

            def update_status(text):
                return text

            output, return_code = self.process_executor.run_pipeline_step(
                cmd,
                cancel_event,
                update_status
            )
            return output
        finally:
            os.chdir(current_dir)


    def run_training(self, project_name: str, tag: str) -> str:
        """Run model training with sanitized tag"""
        if not project_name:
            return "Please select a project"

        safe_tag = tag.replace(':', '').replace(' ', '_')
        cmd = f"python main.py --project {project_name} --train_module standard --tag {safe_tag}"

        current_dir = os.getcwd()
        try:
            os.chdir(self.root_dir)
            self.logger.info(f"Running training command: {cmd}")
            cancel_event = threading.Event()
            self.running_processes[project_name] = cancel_event

            def update_status(text):
                return text

            output, return_code = self.process_executor.run_pipeline_step(
                cmd,
                cancel_event,
                update_status
            )
            return output
        finally:
            os.chdir(current_dir)


    def run_evaluation(self, project_name: str, session_id: str) -> str:
        """Run model evaluation"""
        if not project_name or not session_id:
            return "Please select both project and session"

        cmd = f"python main.py --project {project_name} --eval_module standard --session {session_id}"

        current_dir = os.getcwd()
        try:
            os.chdir(self.root_dir)
            self.logger.info(f"Running evaluation command: {cmd}")
            cancel_event = threading.Event()
            self.running_processes[project_name] = cancel_event

            def update_status(text):
                return text

            output, return_code = self.process_executor.run_pipeline_step(
                cmd,
                cancel_event,
                update_status
            )
            return output
        finally:
            os.chdir(current_dir)


    def get_session_list(self, project_name: str) -> List[str]:
        """Get list of available sessions for a project"""
        if not project_name:
            return []
        try:
            sessions = [d for d in os.listdir(self.project_structure['sessions']) 
                    if os.path.isdir(os.path.join(self.project_structure['sessions'], d)) 
                    and project_name in d]
            return sorted(sessions, reverse=True)
        except Exception as e:
            self.logger.error(f"Error getting session list: {str(e)}")
            return []

    def get_model_results(self, session_id: str) -> Optional[pd.DataFrame]:
        """Get model results for a specific session"""
        if not session_id:
            return None
        try:
            session_dir = os.path.join(self.project_structure['sessions'], session_id)
            models_file = os.path.join(session_dir, 'models.csv')
            if os.path.exists(models_file):
                return pd.read_csv(models_file)
            return None
        except Exception as e:
            self.logger.error(f"Error getting model results: {str(e)}")
            return None

    
    class EDAReport:
        def __init__(self, report_path: str, warnings: List[str]):
            self.report_path = report_path
            self.warnings = warnings

    def generate_eda_report(self, project_name: str, dataset_type: str, vs_criterion: bool = False, force_new: bool = False) -> "GizmoUI.EDAReport":
        """Generate Sweetviz EDA report for the specified dataset"""
        warnings = []
        try:
            import sweetviz as sv
            import numpy as np
            
            if not project_name:
                return self.EDAReport("Please select a project", [])

            # Setup paths
            report_dir = os.path.join(self.project_structure['output_data'], project_name, 'eda')
            os.makedirs(report_dir, exist_ok=True)
            
            report_type = "criterion" if vs_criterion else "all"
            dataset_suffix = "input" if dataset_type == "input" else "processed"
            report_name = f"sweetviz_report_{dataset_suffix}_{report_type}.html"
            report_path = os.path.join(report_dir, report_name)

            # Check existing report unless force_new is True
            if os.path.exists(report_path) and not force_new:
                self.logger.info(f"Using existing report: {report_path}")
                return self.EDAReport(report_path, [])

            def safe_convert_to_numeric(df, column):
                """Safely convert a column to numeric, handling Series boolean ambiguity"""
                try:
                    if str(df[column].dtype) in ['int64', 'float64']:
                        return df[column]
                    
                    # Handle boolean series explicitly
                    if str(df[column].dtype) == 'bool':
                        return df[column].astype(int)
                        
                    # Check if all values are numeric already
                    if pd.to_numeric(df[column], errors='coerce').notna().all():
                        return pd.to_numeric(df[column], errors='coerce')
                        
                    # For datetime columns, convert to Unix timestamp
                    if pd.api.types.is_datetime64_any_dtype(df[column]):
                        warnings.append(f"Converting datetime column {column} to numeric timestamp")
                        return pd.to_numeric(df[column].astype(np.int64) // 10**9)
                        
                    # For categorical/object columns
                    if pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                        # Try to convert directly first
                        numeric_series = pd.to_numeric(df[column], errors='coerce')
                        if numeric_series.notna().any():  # If any conversion succeeded
                            return numeric_series
                        
                        # If direct conversion failed, try label encoding
                        warnings.append(f"Label encoding categorical column {column}")
                        return pd.Categorical(df[column]).codes
                        
                    # Default fallback
                    return pd.to_numeric(df[column], errors='coerce')
                    
                except Exception as e:
                    warnings.append(f"Error converting {column}: {str(e)}")
                    # Return a numeric type series filled with NaN as fallback
                    return pd.Series(np.nan, index=df.index, dtype=float)

            # Load data with warnings
            if dataset_type == 'input':
                input_path = os.path.join(self.project_structure['input_data'], project_name)
                csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
                if not csv_files:
                    return self.EDAReport("No CSV files found in input directory", warnings)
                
                try:
                    print(f"Loading input dataset from {csv_files[0]}")
                    df = pd.read_csv(os.path.join(input_path, csv_files[0]))
                    if len(df) == 0:  # Use len() instead of .empty
                        return self.EDAReport("Dataset is empty", warnings)
                except Exception as e:
                    return self.EDAReport(f"Error reading CSV: {str(e)}", warnings)
                    
            else:  # processed data
                processed_path = os.path.join(self.project_structure['output_data'], project_name, 'output_data_file.parquet')
                processed_full_path = os.path.join(self.project_structure['output_data'], project_name, 'output_data_file_full.parquet')
                
                if not os.path.exists(processed_path) or not os.path.exists(processed_full_path):
                    return self.EDAReport("Processed data files not found. Please run data preparation first.", warnings)
                
                try:
                    df = pd.read_parquet(processed_path)
                    df_full = pd.read_parquet(processed_full_path)
                    if len(df) == 0 or len(df_full) == 0:  # Use len() instead of .empty
                        return self.EDAReport("One or both processed datasets are empty", warnings)
                except Exception as e:
                    return self.EDAReport(f"Error reading parquet files: {str(e)}", warnings)

            # Generate appropriate report
            try:
                if vs_criterion:
                    # Load criterion column from parameters
                    param_path = os.path.join(self.project_structure['params'], f'params_{project_name}.json')
                    if not os.path.exists(param_path):
                        return self.EDAReport("Parameters file not found", warnings)
                    
                    with open(param_path, 'r') as f:
                        params = json.load(f)
                    
                    criterion_col = params.get('criterion_column')
                    if not criterion_col:
                        return self.EDAReport("No criterion column specified in parameters", warnings)
                    
                    if criterion_col not in df.columns:
                        return self.EDAReport(f"Criterion column '{criterion_col}' not found in dataset", warnings)
                    
                    # Safely convert criterion column
                    df[criterion_col] = safe_convert_to_numeric(df, criterion_col)
                    if dataset_type == 'processed':
                        df_full[criterion_col] = safe_convert_to_numeric(df_full, criterion_col)

                    # Check if conversion resulted in all NaN values
                    if df[criterion_col].isna().all():
                        return self.EDAReport(f"Criterion column '{criterion_col}' has no valid numeric values after conversion", warnings)
                    
                    if dataset_type == 'processed':
                        my_report = sv.compare(
                            [df, "Processed Dataset"],
                            [df_full, "Full Dataset"],
                            target_feat=criterion_col,
                            pairwise_analysis='off'
                        )
                    else:
                        my_report = sv.analyze(
                            source=df,
                            target_feat=criterion_col,
                            pairwise_analysis='off'
                        )
                else:
                    if dataset_type == 'processed':
                        my_report = sv.compare(
                            [df, "Processed Dataset"],
                            [df_full, "Full Dataset"],
                            pairwise_analysis='off'
                        )
                    else:
                        my_report = sv.analyze(
                            source=df,
                            pairwise_analysis='off'
                        )

                # Save report
                my_report.show_html(filepath=report_path, open_browser=False)
                return self.EDAReport(report_path, warnings)
                    
            except Exception as e:
                return self.EDAReport(f"Error generating report: {str(e)}", warnings)

        except Exception as e:
            error_msg = f"Error in EDA report generation: {str(e)}"
            self.logger.error(error_msg)
            return self.EDAReport(error_msg, warnings)

    def create_interface(self):
        """Create enhanced Gradio interface with structured parameter form"""
        with gr.Blocks(title="Gizmo UI", css="""
                        .scrollable-report {
                            height: 100vh;  /* Use viewport height */
                            overflow-y: auto;
                            overflow-x: auto;
                            border: 1px solid #ccc;
                            padding: 10px;
                        }
                        .scrollable-report iframe {
                            width: 100%;
                            height: 100vh;  /* Use viewport height */
                            border: none;
                        }
                        """
                        ) as interface:
            gr.Markdown("# Gizmo Machine Learning Pipeline Interface")
            
            with gr.Tab("Project Management"):
                with gr.Row():
                    project_dropdown = gr.Dropdown(
                        choices=self.list_projects(),
                        label="Select Project",
                        interactive=True
                    )
                    refresh_btn = gr.Button("Refresh Projects")
                
                # Create structured form for parameters
                with gr.Group():
                    gr.Markdown("### Project Parameters")
                    criterion_column = gr.Textbox(label="Criterion Column")
                    observation_date = gr.Textbox(label="Observation Date Column")
                    missing_treatment = gr.Radio(
                        choices=["Missing", "column_mean", "median", "delete"],
                        label="Missing Value Treatment",
                        value="Missing"
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Time Periods")
                            t1_period = gr.Textbox(label="T1 Period")
                            t2_period = gr.Textbox(label="T2 Period")
                            t3_period = gr.Textbox(label="T3 Period")
                        
                        with gr.Column():
                            gr.Markdown("### Secondary Criteria")
                            secondary_criteria = gr.Dataframe(
                                headers=["Index", "Criterion"],
                                datatype=["str", "str"],
                                label="Secondary Criterion Columns"
                            )

                    with gr.Accordion("Advanced Settings", open=False):
                        columns_to_exclude = gr.Dataframe(
                            headers=["Column Name"],
                            datatype=["str"],
                            label="Columns to Exclude"
                        )
                        periods_to_exclude = gr.Dataframe(
                            headers=["Period"],
                            datatype=["str"],
                            label="Periods to Exclude"
                        )

                save_params_btn = gr.Button("Save Parameters", variant="primary")
                save_status = gr.Textbox(label="Save Status")

            with gr.Tab("EDA"):
                with gr.Row():
                    dataset_type = gr.Radio(
                        choices=["input", "processed"],
                        label="Select Dataset",
                        value="input"
                    )
                
                with gr.Row():
                    analyze_btn = gr.Button("Load Report", variant="secondary")
                    run_btn = gr.Button("Run New Analysis", variant="primary")
                    analyze_criterion_btn = gr.Button("Load Criterion Report", variant="secondary")
                    run_criterion_btn = gr.Button("Run New Criterion Analysis", variant="primary")
                
                # Add warnings display
                warnings_box = gr.Markdown(
                    label="Data Processing Warnings",
                    value="",
                    visible=False
                )
                
                # Output HTML iframe for the report
                report_html = gr.HTML(
                    label="Analysis Report",
                    value="", 
                    elem_classes="scrollable-report"
                )
                
                # In create_interface method:

                # Update the process_eda_result function's HTML handling:
                def process_eda_result(project, dataset, vs_criterion, force_new=False):
                    """Process EDA report generation with proper handling of new vs existing reports"""
                    print(f"Processing EDA report: project={project}, dataset={dataset}, vs_criterion={vs_criterion}, force_new={force_new}")
                    
                    result = self.generate_eda_report(project, dataset, vs_criterion=vs_criterion, force_new=force_new)
                    
                    warnings_html = ""
                    if result.warnings:
                        warnings_html = "### Data Processing Warnings:\n" + "\n".join([f"- {w}" for w in result.warnings])
                        warnings_box.visible = True
                    else:
                        warnings_box.visible = False
                    
                    if result.report_path.endswith('.html'):
                        with open(result.report_path, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                            iframe_content = f'''
                            <div style="width:100%;height:100vh">
                                <iframe srcdoc="{report_content.replace('"', '&quot;')}" 
                                        style="width:100%;height:100vh;border:none;">
                                </iframe>
                            </div>
                            '''
                    else:
                        iframe_content = f"<div class='error'>{result.report_path}</div>"
                    
                    return warnings_html, iframe_content

                # Button handlers - Using consistent naming and direct function calls
                run_btn.click(
                    fn=process_eda_result,
                    inputs=[
                        project_dropdown,
                        dataset_type,
                        gr.Checkbox(value=False, visible=False),  # vs_criterion
                        gr.Checkbox(value=True, visible=False)    # force_new
                    ],
                    outputs=[warnings_box, report_html]
                )

                run_criterion_btn.click(
                    fn=process_eda_result,
                    inputs=[
                        project_dropdown,
                        dataset_type,
                        gr.Checkbox(value=True, visible=False),   # vs_criterion
                        gr.Checkbox(value=True, visible=False)    # force_new
                    ],
                    outputs=[warnings_box, report_html]
                )

                analyze_btn.click(
                    fn=process_eda_result,
                    inputs=[
                        project_dropdown,
                        dataset_type,
                        gr.Checkbox(value=False, visible=False),  # vs_criterion
                        gr.Checkbox(value=False, visible=False)   # force_new
                    ],
                    outputs=[warnings_box, report_html]
                )

                analyze_criterion_btn.click(
                    fn=process_eda_result,
                    inputs=[
                        project_dropdown,
                        dataset_type,
                        gr.Checkbox(value=True, visible=False),   # vs_criterion
                        gr.Checkbox(value=False, visible=False)   # force_new
                    ],
                    outputs=[warnings_box, report_html]
                )


            with gr.Tab("Pipeline Execution"):
                with gr.Row():
                    tag_input = gr.Textbox(
                        label="Run Tag",
                        value=datetime.now().strftime("%Y%m%d_%H%M%S"),  # Changed format to avoid colons
                        placeholder="Enter a tag without special characters"
                    )
                
                with gr.Row():
                    data_prep_btn = gr.Button("Run Data Preparation")
                    train_btn = gr.Button("Run Training")
                    eval_btn = gr.Button("Run Evaluation")
                
                progress = gr.Progress()
                status = gr.Textbox(label="Status")
                cancel_btn = gr.Button("Cancel Operation", variant="stop")

            with gr.Tab("Results"):
                with gr.Row():
                    session_dropdown = gr.Dropdown(
                        label="Select Session",
                        interactive=True
                    )
                    refresh_sessions_btn = gr.Button("Refresh Sessions")
                
                results_df = gr.DataFrame(label="Model Results")

            # Event handlers
            def on_analyze(project, dataset, vs_criterion):
                result = self.generate_eda_report(project, dataset, vs_criterion)
                if result.report_path.endswith('.html'):
                    with open(result.report_path, 'r', encoding='utf-8') as f:
                        return f.read()
                return f"<div class='error'>{result.report_path}</div>"
            
            analyze_btn.click(
                fn=on_analyze,
                inputs=[project_dropdown, dataset_type, gr.Checkbox(value=False, visible=False)],
                outputs=report_html
            )
            
            analyze_criterion_btn.click(
                fn=on_analyze,
                inputs=[project_dropdown, dataset_type, gr.Checkbox(value=True, visible=False)],
                outputs=report_html
            )

            def on_analyze_existing(project, dataset, vs_criterion):
                report_dir = os.path.join(self.project_structure['output_data'], project, 'eda')
                report_type = "criterion" if vs_criterion else "all"
                dataset_suffix = "input" if dataset == "input" else "processed"
                report_name = f"sweetviz_report_{dataset_suffix}_{report_type}.html"
                report_path = os.path.join(report_dir, report_name)
                
                if os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        return f.read()
                return f"<div class='error'>No existing report found. Please run a new analysis.</div>"
            
            def refresh_projects():
                return gr.Dropdown(choices=self.list_projects())

            

            project_dropdown.change(
                fn=self.load_param_form,
                inputs=[project_dropdown],
                outputs=[
                    criterion_column,
                    observation_date,
                    missing_treatment,
                    t1_period,
                    t2_period,
                    t3_period,
                    secondary_criteria,
                    columns_to_exclude,
                    periods_to_exclude
                ]
            )

            refresh_btn.click(
                fn=refresh_projects,
                outputs=[project_dropdown]
            )

            save_params_btn.click(
                fn=self.save_param_form,
                inputs=[
                    project_dropdown,
                    criterion_column,
                    observation_date,
                    missing_treatment,
                    t1_period,
                    t2_period,
                    t3_period,
                    secondary_criteria,
                    columns_to_exclude,
                    periods_to_exclude
                ],
                outputs=[save_status]
            )

            status = gr.Textbox(
                label="Status", 
                interactive=False,
                lines=15,  # Show more lines
                autoscroll=True,  # Automatically scroll to bottom
                show_copy_button=True  # Allow copying the output
            )
            
            data_prep_btn.click(
                fn=self.run_data_prep,
                inputs=[project_dropdown, tag_input],
                outputs=status,
                show_progress=True
            )


            train_btn.click(
                fn=self.run_training,
                inputs=[project_dropdown, tag_input],
                outputs=status,
                show_progress=True
            )

            eval_btn.click(
                fn=self.run_evaluation,
                inputs=[project_dropdown, session_dropdown],
                outputs=status,
                show_progress=True
            )

            cancel_btn.click(
                fn=lambda x: self.running_processes.get(x, threading.Event()).set() if x else None,
                inputs=[project_dropdown]
            )

            refresh_sessions_btn.click(
                fn=self.get_session_list,
                inputs=[project_dropdown],
                outputs=[session_dropdown]
            )

            session_dropdown.change(
                fn=self.get_model_results,
                inputs=[session_dropdown],
                outputs=[results_df]
            )

        return interface

if __name__ == "__main__":
    ui = GizmoUI()
    interface = ui.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )