import os

def generate_version(path_to_dir, new_version=True):
    """Create a new folder contains new_version weights file

    Returns new_version, path to saving folder and path to save weights file (preferably .hdf5 files)

    Keyword arguments:

    path_to_dir -- path to directory where new folder should be created

    new_version -- version of the saved model and new weights file (default to 1)
    """

    assert(type(new_version) == bool) , "new_version must be integer"

    versions_list = sorted([int(version.removeprefix('version')) for version in os.listdir(path_to_dir) if version != ".DS_Store"])

    if new_version == True:
        new_version = versions_list[-1] + 1 if len(versions_list) > 0 else 1
    else:
        new_version = versions_list[-1] if len(versions_list) > 0 else 1

    new_version_folder = 'version{}'.format(int(new_version))
    new_version_file = 'weights.best.hdf5'

    path = os.path.join(path_to_dir, new_version_folder)
    save_path = os.path.join(path, new_version_file)
    if versions_list.count(new_version) == 0:
        os.makedirs(path)

    return new_version, path, save_path