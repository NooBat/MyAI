import os

def generate_version(path_to_dir, new_file_name='', new_version=True):
    """Create a new folder contains new_version weights file

    Returns new_version, path to saving folder and path to save new_file_name (preferably .hdf5 files)

    Keyword arguments:

    path_to_dir -- path to directory where new folder should be created

    new_file_name -- name of the desired file (default to '')

    new_version -- version of the saved model and new weights file (default to 1)
    """

    assert(type(new_version) == bool) , "new_version must be integer"

    versions_list = sorted([int(version.removeprefix('version')) for version in os.listdir(path_to_dir) if version != ".DS_Store"])

    if new_version == True:
        new_version = versions_list[-1] + 1 if len(versions_list) > 0 else 1
    else:
        new_version = versions_list[-1] if len(versions_list) > 0 else 1

    new_version_folder = 'version{}'.format(int(new_version))
    new_version_file = new_file_name if len(new_file_name) > 0 else None

    save_path = path = os.path.join(path_to_dir, new_version_folder)
    if new_version_file is not None:
        save_path = os.path.join(save_path, new_version_file)
    if versions_list.count(new_version) == 0:
        os.makedirs(path)

    return new_version, path, save_path