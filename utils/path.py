import os
import os.path

def mkdir_if_not_exist(dir_name, mode=0o777):
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
    