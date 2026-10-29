import os
import tempfile
import unittest
from unittest import mock

from nnunetv2.model_sharing import model_download


class _BrokenOffResponse:
    """A streamed response whose connection breaks after the first chunk."""
    headers = {'Content-Length': '8'}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b'1234'
        raise ConnectionError('connection lost')


class TestDownloadFile(unittest.TestCase):
    def test_file_is_closed_when_the_download_fails(self):
        opened = []

        def recording_open(*args, **kwargs):
            f = open(*args, **kwargs)
            opened.append(f)
            return f

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(model_download.requests, 'get', return_value=_BrokenOffResponse()), \
                mock.patch.object(model_download, 'open', recording_open, create=True):
            try:
                model_download.download_file('https://example.com/model.zip', os.path.join(tmp, 'model.zip'))
            except ConnectionError:
                # while the error propagates, its traceback keeps download_file's locals alive. Only an explicit
                # close releases the file then, and on Windows the caller cannot delete an open file
                self.assertEqual([f.closed for f in opened], [True])
            else:
                self.fail('the download did not fail')


if __name__ == '__main__':
    unittest.main()
