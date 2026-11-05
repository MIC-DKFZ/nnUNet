import sys
import unittest
from unittest import mock

from nnunetv2.model_sharing import entry_points


class TestExportEntryPointStrict(unittest.TestCase):
    def _strict_for(self, extra_args):
        argv = ['nnUNetv2_export_model_to_zip', '-d', '4', '-o', 'model.zip'] + extra_args
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(entry_points, 'export_pretrained_model') as export:
            entry_points.export_pretrained_model_entry()
        export.assert_called_once()
        return export.call_args.kwargs['strict']

    def test_strict_by_default(self):
        self.assertIs(self._strict_for([]), True)

    def test_not_strict_flag_disables_strict(self):
        self.assertIs(self._strict_for(['--not_strict']), False)


if __name__ == '__main__':
    unittest.main()
