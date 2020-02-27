from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.utils import split_4d


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The MSD provides data as 4D Niftis with the modality being the first"
                                                 " dimension. We think this may be cumbersome for some users and "
                                                 "therefore expect 3D niftixs instead, with one file per modality. "
                                                 "This utility will convert 4D MSD data into the format nnU-Net "
                                                 "expects")
    parser.add_argument("-i", help="Input folder. Must point to a TaskXX_TASKNAME folder as downloaded from the MSD "
                                   "website", required=True)
    parser.add_argument("-p", required=False, default=default_num_threads, type=int,
                        help="Use this to specify how many processes are used to run the script. "
                             "Default is %d" % default_num_threads)
    args = parser.parse_args()

    split_4d(args.i, args.p)
