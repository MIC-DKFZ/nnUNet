from batchgenerators.utilities.file_and_folder_operations import *
import os


def recursive_rename(folder):
    s = subdirs(folder, join=False)
    for ss in s:
        if ss.startswith("Task") and ss.find("_") == 6:
            task_id = int(ss[4:6])
            name = ss[7:]
            os.rename(join(folder, ss), join(folder, "Task%03.0d_" % task_id + name))
    s = subdirs(folder, join=True)
    for ss in s:
        recursive_rename(ss)
