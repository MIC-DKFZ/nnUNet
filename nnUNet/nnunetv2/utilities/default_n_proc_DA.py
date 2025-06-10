import subprocess
import os


def get_allowed_n_proc_DA():
    """
    This function is used to set the number of processes used on different Systems. It is specific to our cluster
    infrastructure at DKFZ. You can modify it to suit your needs. Everything is allowed.

    IMPORTANT: if the environment variable nnUNet_n_proc_DA is set it will overwrite anything in this script
    (see first line).

    Interpret the output as the number of processes used for data augmentation PER GPU.

    The way it is implemented here is simply a look up table. We know the hostnames, CPU and GPU configurations of our
    systems and set the numbers accordingly. For example, a system with 4 GPUs and 48 threads can use 12 threads per
    GPU without overloading the CPU (technically 11 because we have a main process as well), so that's what we use.
    """

    if 'nnUNet_n_proc_DA' in os.environ.keys():
        use_this = int(os.environ['nnUNet_n_proc_DA'])
    else:
        hostname = subprocess.getoutput(['hostname'])
        if hostname in ['Fabian', 'isensee-']:
            use_this = 12
        elif hostname in ['hdf19-gpu16', 'hdf19-gpu17', 'hdf19-gpu18', 'hdf19-gpu19', 'e230-AMDworkstation']:
            use_this = 16
        elif hostname.startswith('e230-dgx1'):
            use_this = 10
        elif hostname.startswith('hdf18-gpu') or hostname.startswith('e132-comp'):
            use_this = 16
        elif hostname.startswith('e230-dgx2'):
            use_this = 6
        elif hostname.startswith('e230-dgxa100-'):
            use_this = 28
        elif hostname.startswith('e230-thinka100-'):
            use_this = 20
        elif hostname.startswith('lsf22-gpu'):
            use_this = 28
        elif hostname.startswith('hdf19-gpu') or hostname.startswith('e071-gpu'):
            use_this = 12
        else:
            use_this = 12  # default value

    use_this = min(use_this, os.cpu_count())
    return use_this
