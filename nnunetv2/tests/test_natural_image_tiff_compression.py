import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
from skimage import io

from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO


class TestNaturalImage2DIOTiffCompression(unittest.TestCase):
    def test_supported_file_endings_include_tiff(self):
        self.assertIn('.tif', NaturalImage2DIO.supported_file_endings)
        self.assertIn('.tiff', NaturalImage2DIO.supported_file_endings)

    def test_sparse_tif_is_packbits_roundtrips_and_is_smaller_than_uncompressed(self):
        rw = NaturalImage2DIO()
        seg = _sparse_label_map(shape=(1, 1024, 1024), seed=0)
        expected = seg[0]

        with TemporaryDirectory() as tmp:
            compressed = os.path.join(tmp, 'pred.tif')
            uncompressed = os.path.join(tmp, 'uncompressed.tif')
            rw.write_seg(seg, compressed, {})
            tifffile.imwrite(uncompressed, expected)

            np.testing.assert_array_equal(tifffile.imread(compressed), expected)
            read_back, props = rw.read_seg(compressed)
            np.testing.assert_array_equal(read_back[0, 0], expected)
            self.assertEqual(props['spacing'], (999, 1, 1))

            with tifffile.TiffFile(compressed) as tif:
                self.assertEqual(tif.pages[0].compression, tifffile.COMPRESSION.PACKBITS)
                self.assertEqual(tif.pages[0].dtype, np.uint8)

            compressed_size = os.path.getsize(compressed)
            uncompressed_size = os.path.getsize(uncompressed)
            self.assertLess(compressed_size * 10, uncompressed_size)
            self.assertLess(compressed_size, expected.nbytes)

    def test_tiff_extension_and_uppercase_use_packbits(self):
        rw = NaturalImage2DIO()
        seg = _sparse_label_map(shape=(1, 256, 256), seed=1)
        expected = seg[0]

        with TemporaryDirectory() as tmp:
            for name in ('pred.tiff', 'pred.TIF', 'pred.TIFF'):
                out = os.path.join(tmp, name)
                rw.write_seg(seg, out, {})
                np.testing.assert_array_equal(tifffile.imread(out), expected)
                with tifffile.TiffFile(out) as tif:
                    self.assertEqual(tif.pages[0].compression, tifffile.COMPRESSION.PACKBITS, name)

    def test_labels_at_or_above_255_are_uint16(self):
        rw = NaturalImage2DIO()
        seg = np.zeros((1, 32, 32), dtype=np.int32)
        seg[0, 1, 2] = 255

        with TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'wide.tif')
            rw.write_seg(seg, out, {})
            written = tifffile.imread(out)
            self.assertEqual(written.dtype, np.uint16)
            np.testing.assert_array_equal(written, seg[0].astype(np.uint16))

    def test_png_write_is_unchanged(self):
        rw = NaturalImage2DIO()
        seg = np.zeros((1, 16, 16), dtype=np.uint8)
        seg[0, 2:6, 3:8] = 4

        with TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'pred.png')
            rw.write_seg(seg, out, {})
            np.testing.assert_array_equal(io.imread(out), seg[0])
            with open(out, 'rb') as f:
                self.assertEqual(f.read(8), b'\x89PNG\r\n\x1a\n')


def _sparse_label_map(shape, seed):
    rng = np.random.RandomState(seed)
    seg = np.zeros(shape, dtype=np.uint8)
    n = 32
    ys = rng.randint(0, shape[1], size=n)
    xs = rng.randint(0, shape[2], size=n)
    seg[0, ys, xs] = rng.randint(1, 8, size=n)
    return seg


if __name__ == '__main__':
    unittest.main()
