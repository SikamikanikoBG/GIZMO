import os
import pandas as pd

class CrossValDataLoader:
    def __init__(self, data_dir: str):
        """
        CrossValDataLoader - A class for loading data from a directory into Pandas DataFrames.

        This class streamlines the process of loading multiple files from a single directory into
        Pandas DataFrames for cross-validation or other data analysis tasks. It supports various file
        formats and dynamically sets DataFrame attributes based on file names.

        Attributes:
            data_dir (str): The path to the directory containing the data files.

        Methods:
            load_files(): Loads all supported files (CSV, PKL, Feather, Parquet) into Pandas DataFrames.
                         DataFrames are named after their filenames (without extension) and set as class attributes.
        """

        self.data_dir = data_dir

        self.load_files()

    def load_files(self):
        for filename in os.listdir(self.data_dir):
            name, extension = os.path.splitext(filename)
            file_path = os.path.join(self.data_dir, filename)

            try:
                if extension == '.csv':
                    setattr(self, name, pd.read_csv(file_path))
                elif extension == '.pkl':
                    setattr(self, name, pd.read_pickle(file_path))
                elif extension == '.feather':
                    setattr(self, name, pd.read_feather(file_path))
                elif extension == '.parquet':
                    setattr(self, name, pd.read_parquet(file_path))
                else:
                    print(f"Skipping unsupported file: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")