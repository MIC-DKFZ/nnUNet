#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.utils import split_4d


def main():
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
    parser.add_argument("-output_task_id", required=False, default=None, type=int,
                        help="If specified, this will overwrite the task id in the output folder. If unspecified, the "
                             "task id of the input folder will be used.")
    args = parser.parse_args()

    split_4d(args.i, args.p)


if __name__ == "__main__":
    main()
