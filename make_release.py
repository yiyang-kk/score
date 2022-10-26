# make_release.py
import argparse
from os import listdir, makedirs, path
from shutil import copyfile, move, copytree, rmtree
from datetime import datetime
from scoring import __version__ as VERSION

WORKFLOW_DIRS = ["workflow"]
PACKAGE_DIRS = ["scoring"]
DIRS_TO_ARCHIVE = [r"workflow/config_data_creator" ,r"workflow/demo_data", r"workflow/coll_demo_data"]

def get_today_string():
    """Generates current date in YYYYMMDD format.
    
    Returns:
        str: Current date. 
    """

    # format date to YYYYMMDD (single digits are zero-padded)
    return datetime.today().strftime("%Y%m%d")


def archive_versioned(directory, version, archive_location):
    """Creates a zip archive of specified `directory` in
    archive location. Archive name has appended supplied `version`
    and today's timestamp.
    
    Args:
        directory (str): relative path to directory to be archived
        version (str): in X.Y.Z format
        archive_location (str): relative path to directory
                                for resulting archive
    
    Returns:
        str: relative path to created archive
    """
    from shutil import make_archive

    base_dir = path.basename(directory)

    version = version.replace(".", "_")
    archive_name = path.join(archive_location, f"{base_dir}_{version}")
    make_archive(archive_name, "zip", directory)
    return f"{archive_name}.zip"


def find_all_notebooks(directory):
    """Finds all `.ipynb` file in specified directory.
    Doesn't search inside sub-directories
    
    Args:
        directory ([str]): relative/absolute path to directory
    
    Returns:
        [list(str)]: of relative/absolute paths to found files 
    """
    # looks just inside directory, doesn't search sub-directories
    return [path.join(directory, filename) for filename in listdir(directory) if filename.endswith(".ipynb")]

def main():
    # parse arguments, only one positional required `version`

    WORKFLOW_FILES = []


    # check if all supplied folders exist and find all notebooks
    for workflow_dir in WORKFLOW_DIRS:
        if path.exists(workflow_dir):
            WORKFLOW_FILES.extend(find_all_notebooks(workflow_dir))
        else:
            raise FileNotFoundError(f"'{workflow_dir}' folder not found.")

    for dir_ in DIRS_TO_ARCHIVE + PACKAGE_DIRS:
        if not path.exists(dir_):
            raise FileNotFoundError(f"Folder {dir_} does not exists. Exiting")


    # print folders and notebooks for user to check
    print("Package folders to be archived:")
    for dir_ in PACKAGE_DIRS:
        print(f"  {dir_}")

    print("\nDemo data folders to be archived:")
    for dir_ in DIRS_TO_ARCHIVE:
        print(f"  {dir_}")

    print("\nWorkflows found:")
    for ntb in sorted(WORKFLOW_FILES):
        print(f"  {ntb}")

    # create release folder
    RELEASE_DIR = f"""release"""
    if not path.exists(RELEASE_DIR):
        makedirs(RELEASE_DIR)
    else:
        raise Exception(f"Folder `{RELEASE_DIR}` already exists. Exiting")


    # copy workflows to temp dir
    WORKFLOW_DIR_TMP = path.join(RELEASE_DIR, "workflows")
    makedirs(WORKFLOW_DIR_TMP, exist_ok=True)

    for ntb in WORKFLOW_FILES:
        copyfile(path.join(ntb), path.join(WORKFLOW_DIR_TMP, path.basename(ntb)))


    ## copy demo data to temp dir
    DEMO_DATA_DIR_TMP = path.join(RELEASE_DIR, "demo_data")
    makedirs(DEMO_DATA_DIR_TMP, exist_ok=True)

    for dir_ in DIRS_TO_ARCHIVE:
        copytree(dir_, path.join(DEMO_DATA_DIR_TMP, path.basename(dir_)))

    ## copy demo data to temp dir
    PACKAGE_DIR_TMP = path.join(RELEASE_DIR, "scoring")
    makedirs(PACKAGE_DIR_TMP, exist_ok=True)

    for dir_ in PACKAGE_DIRS:
        copytree(dir_, path.join(PACKAGE_DIR_TMP, path.basename(dir_)))

    ## create archives
    archive_versioned(DEMO_DATA_DIR_TMP, VERSION, RELEASE_DIR)
    archive_versioned(WORKFLOW_DIR_TMP, VERSION, RELEASE_DIR)
    archive_versioned(PACKAGE_DIR_TMP, VERSION, RELEASE_DIR)

    # clean up temp folders
    rmtree(WORKFLOW_DIR_TMP)
    rmtree(DEMO_DATA_DIR_TMP)
    rmtree(PACKAGE_DIR_TMP)

    # print(f"Created {len(DIRS_TO_ARCHIVE)} archives and copied {len(WORKFLOW_FILES)} notebooks to `{RELEASE_DIR}`")


if __name__ == "__main__":
    main()
