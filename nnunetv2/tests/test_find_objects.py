import os
import unittest
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest.mock import patch

from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


def _write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(dedent(content))


class TestFindObjects(unittest.TestCase):
    def test_external_trainer_lookup_handles_multiple_directories_with_same_package_name(self):
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            os.makedirs(os.path.join(first_dir, "sharedpkg"))
            os.makedirs(os.path.join(second_dir, "sharedpkg"))

            _write_file(os.path.join(first_dir, "sharedpkg", "__init__.py"), "")
            _write_file(
                os.path.join(first_dir, "sharedpkg", "trainer.py"),
                """
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


                class FirstTrainer(nnUNetTrainer):
                    pass
                """,
            )

            _write_file(os.path.join(second_dir, "sharedpkg", "__init__.py"), "")
            _write_file(
                os.path.join(second_dir, "sharedpkg", "trainer.py"),
                """
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


                class SecondTrainer(nnUNetTrainer):
                    pass
                """,
            )

            with patch.dict(
                os.environ,
                {"nnUNet_extTrainer": os.pathsep.join((first_dir, second_dir))},
                clear=False,
            ):
                trainer_class = recursive_find_trainer_class_by_name("SecondTrainer")

            self.assertEqual(trainer_class.__name__, "SecondTrainer")

    def test_external_trainer_lookup_surfaces_import_errors(self):
        with TemporaryDirectory() as trainer_dir:
            os.makedirs(os.path.join(trainer_dir, "brokenpkg"))

            _write_file(os.path.join(trainer_dir, "brokenpkg", "__init__.py"), "")
            _write_file(
                os.path.join(trainer_dir, "brokenpkg", "trainer.py"),
                """
                import definitely_missing_dependency
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


                class BrokenTrainer(nnUNetTrainer):
                    pass
                """,
            )

            with patch.dict(
                os.environ,
                {"nnUNet_extTrainer": trainer_dir},
                clear=False,
            ):
                with self.assertRaises(ModuleNotFoundError) as exc:
                    recursive_find_trainer_class_by_name("BrokenTrainer")

            self.assertIn("definitely_missing_dependency", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
