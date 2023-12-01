from pathlib import Path

from nnunetv2.model_sharing.model_download import download_and_install_from_url
from nnunetv2.model_sharing.model_export import export_pretrained_model
from nnunetv2.model_sharing.model_import import install_model_from_zip_file
from nnunetv2.model_sharing.onnx_export import export_onnx_model


def print_license_warning():
    print("")
    print("######################################################")
    print("!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!")
    print("######################################################")
    print(
        "Using the pretrained model weights is subject to the license of the dataset they were trained on. Some "
        "allow commercial use, others don't. It is your responsibility to make sure you use them appropriately! Use "
        "nnUNet_print_pretrained_model_info(task_name) to see a summary of the dataset and where to find its license!"
    )
    print("######################################################")
    print("")


def download_by_url():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use this to download pretrained models. This script is intended to download models via url only. "
        "CAREFUL: This script will overwrite "
        "existing models (if they share the same trainer class and plans as "
        "the pretrained model."
    )
    parser.add_argument("url", type=str, help="URL of the pretrained model")
    args = parser.parse_args()
    url = args.url
    download_and_install_from_url(url)


def install_from_zip_entry_point():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use this to install a zip file containing a pretrained model."
    )
    parser.add_argument("zip", type=str, help="zip file")
    args = parser.parse_args()
    zip = args.zip
    install_model_from_zip_file(zip)


def export_pretrained_model_entry():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use this to export a trained model as a zip file."
    )
    parser.add_argument("-d", type=str, required=True, help="Dataset name or id")
    parser.add_argument("-o", type=str, required=True, help="Output file name")
    parser.add_argument(
        "-c",
        nargs="+",
        type=str,
        required=False,
        default=("3d_lowres", "3d_fullres", "2d", "3d_cascade_fullres"),
        help="List of configuration names",
    )
    parser.add_argument(
        "-tr", required=False, type=str, default="nnUNetTrainer", help="Trainer class"
    )
    parser.add_argument(
        "-p", required=False, type=str, default="nnUNetPlans", help="plans identifier"
    )
    parser.add_argument(
        "-f",
        required=False,
        nargs="+",
        type=str,
        default=(0, 1, 2, 3, 4),
        help="list of fold ids",
    )
    parser.add_argument(
        "-chk",
        required=False,
        nargs="+",
        type=str,
        default=("checkpoint_final.pth",),
        help="List of checkpoint names to export. Default: checkpoint_final.pth",
    )
    parser.add_argument(
        "--not_strict",
        action="store_false",
        default=False,
        required=False,
        help="Set this to allow missing folds and/or configurations",
    )
    parser.add_argument(
        "--exp_cv_preds",
        action="store_true",
        required=False,
        help="Set this to export the cross-validation predictions as well",
    )
    args = parser.parse_args()

    export_pretrained_model(
        dataset_name_or_id=args.d,
        output_file=args.o,
        configurations=args.c,
        trainer=args.tr,
        plans_identifier=args.p,
        folds=args.f,
        strict=not args.not_strict,
        save_checkpoints=args.chk,
        export_crossval_predictions=args.exp_cv_preds,
    )


def export_pretrained_model_onnx_entry():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use this to export a trained model to ONNX format."
        "You are responsible for creating the ONNX pipeline yourself."
    )
    parser.add_argument("-d", type=str, required=True, help="Dataset name or id")
    parser.add_argument("-o", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "-c",
        nargs="+",
        type=str,
        required=False,
        default=("3d_lowres", "3d_fullres", "2d", "3d_cascade_fullres"),
        help="List of configuration names",
    )
    parser.add_argument(
        "-tr", required=False, type=str, default="nnUNetTrainer", help="Trainer class"
    )
    parser.add_argument(
        "-p", required=False, type=str, default="nnUNetPlans", help="plans identifier"
    )
    parser.add_argument(
        "-f", required=False, nargs="+", type=str, default=None, help="list of fold ids"
    )
    parser.add_argument(
        "-chk",
        required=False,
        nargs="+",
        type=str,
        default=("checkpoint_final.pth",),
        help="List of checkpoint names to export. Default: checkpoint_final.pth",
    )
    parser.add_argument(
        "--not_strict",
        action="store_false",
        default=False,
        required=False,
        help="Set this to allow missing folds and/or configurations",
    )
    parser.add_argument(
        "--exp_cv_preds",
        action="store_true",
        required=False,
        help="Set this to export the cross-validation predictions as well",
    )
    parser.add_argument(
        "-v",
        action="store_false",
        default=False,
        required=False,
        help="Set this to get verbose output",
    )
    args = parser.parse_args()

    print("######################################################")
    print("!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!")
    print("######################################################")
    print(
        "You are responsible for creating the ONNX pipeline\n"
        "yourself.\n\n"
        "This script will only export the model weights to\n"
        "an onnx file, and some basic information about\n"
        "the model. You will have to create the ONNX pipeline\n"
        "yourself.\n"
    )
    print(
        "See\n"
        "https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html"
        "#execute-the-onnx-model-with-onnx-runtime\n"
        "for some documentation on how to do this."
    )
    print("######################################################")
    print("")

    export_onnx_model(
        dataset_name_or_id=args.d,
        output_dir=args.o,
        configurations=args.c,
        trainer=args.tr,
        plans_identifier=args.p,
        folds=args.f,
        strict=not args.not_strict,
        save_checkpoints=args.chk,
        verbose=args.v,
    )
