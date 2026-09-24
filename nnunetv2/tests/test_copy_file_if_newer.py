import os
import unittest
from tempfile import TemporaryDirectory

from nnunetv2.utilities.file_path_utilities import copy_file_if_newer


class TestCopyFileIfNewer(unittest.TestCase):
    def _write(self, path: str, content: str, mtime: float) -> None:
        with open(path, 'w') as f:
            f.write(content)
        os.utime(path, (mtime, mtime))

    def _read(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def test_copies_when_destination_is_missing(self):
        with TemporaryDirectory() as tmp:
            source, destination = os.path.join(tmp, 'source'), os.path.join(tmp, 'destination')
            self._write(source, 'label', 1_000_000)
            copy_file_if_newer(source, destination)
            self.assertEqual(self._read(destination), 'label')
            # timestamps are preserved, like distutils.file_util.copy_file did
            self.assertEqual(os.path.getmtime(destination), 1_000_000)

    def test_skips_when_destination_is_up_to_date(self):
        with TemporaryDirectory() as tmp:
            source, destination = os.path.join(tmp, 'source'), os.path.join(tmp, 'destination')
            self._write(source, 'new', 1_000_000)
            self._write(destination, 'existing', 1_000_000)
            copy_file_if_newer(source, destination)
            self.assertEqual(self._read(destination), 'existing')

    def test_copies_when_source_is_newer(self):
        with TemporaryDirectory() as tmp:
            source, destination = os.path.join(tmp, 'source'), os.path.join(tmp, 'destination')
            self._write(destination, 'old', 1_000_000)
            self._write(source, 'new', 2_000_000)
            copy_file_if_newer(source, destination)
            self.assertEqual(self._read(destination), 'new')


if __name__ == '__main__':
    unittest.main()
