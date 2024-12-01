from typing import Mapping
import shutil
import anndata 
import os

    
class Dataset:
    def __init__(self, adata_list : Mapping[str, anndata.AnnData ]) -> None:
        self.adata_list = adata_list
            

def create_model_directories(base_path, subdirs):

    if os.path.isabs(base_path):
        try:
            if os.path.exists(base_path):
                shutil.rmtree(base_path)
                print(f"Deleted existing directory: {base_path}")
                
            os.makedirs(base_path)
            print(f"Model directory created: {base_path}")
            for subdir in subdirs:
                subdir_path = os.path.join(base_path, subdir)
                os.makedirs(subdir_path)

        except Exception as e:
            print(f"Error creating directories: {e}")
    else:
        print(f"Invalid path: {base_path} is not valid path.")

