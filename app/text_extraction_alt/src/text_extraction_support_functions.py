import os
import shutil
from os.path import join
import numpy as np

# %% FILE IO Support Functions
def find_files_recursively(directory):
    file_list = []
    dir_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))

        for dir in dirs:
            dir_list.append(os.path.join(root, dir))

    return file_list, dir_list


def delete_contents_in_directory(directory_path, verbose=False):
    try:
        # Delete all contents in the directory recursively
        # shutil.rmtree(directory_path)
        # os.makedirs(directory_path)

        #NOTE: Just using rmtree sometimes produces an error where 
        # sub-directories cannot be removed if stuff is in them.  
        # Deleting the files 1st seems to help.

        files, dirs = find_files_recursively(directory_path)

        for f in files:
            os.remove(f)

        for d in dirs:
            shutil.rmtree(d)

        if verbose:
            print(f'All contents in "{directory_path}" have been deleted successfully.')
    except Exception as e:
        print(f"An error occurred: {e}")
        raise Exception(e)


def reset_directory(directory_path, erase_contents=False, verbose=False):
    """
    Checks if a directory exists.
        - If it does it can erase all condents.
        - If it does not, it will create it.
    """
    try:
        if os.path.exists(directory_path):
            if verbose:
                print(f"Directory: {directory_path}, was found.")
            if erase_contents:
                delete_contents_in_directory(directory_path, verbose)
        else:
            os.makedirs(directory_path)
            if verbose:
                print(f"Directory: {directory_path}, was created.")
    except Exception as e:
        print(f"Unable to clear directory: {directory_path}")
        raise Exception(e)


def get_filename_without_extension(file_path):
    # Get the base name of the file without extension
    base_name = os.path.basename(file_path)
    # Get the filename without extension
    filename_without_extension = os.path.splitext(base_name)[0]
    return filename_without_extension


def list_files_with_extensions(folder_path, extensions):
    file_list = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(extension.lower()) for extension in extensions):
            file_list.append(join(folder_path, file))

    return np.sort(np.array(file_list))