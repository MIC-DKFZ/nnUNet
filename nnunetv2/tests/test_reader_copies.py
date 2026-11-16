"""
NibabelIO(WithReorient) and SimpleITKIO(WithReorient) read an image without the full-size copies they used to make on the
way (nibabel: a float64 image plus a scaling temporary; SimpleITK: a numpy copy of every ITK buffer). The tests pin both
halves: the readers return exactly what the straightforward implementation returns (the _reference_* functions below are
the readers before the change), and their peak memory stays near the stored image plus the float32 result.
"""
import gzip
import os
import shutil
import subprocess
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

import nibabel
import numpy as np
import SimpleITK as sitk
from nibabel.orientations import io_orientation

from nnunetv2.imageio import nibabel_reader_writer
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO, SimpleITKIOWithReorient


def _rotation_z(degrees: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


# direction matrices (columns = where the storage axes point, RAS+ world as in NIfTI): identity (RAS), the common CT
# storages LAS and LPS, axes in permuted order, and an oblique one
DIRECTIONS = {
    'RAS': np.eye(3),
    'LAS': np.diag([-1., 1, 1]),
    'LPS': np.diag([-1., -1, 1]),
    'SRA_permuted': np.array([[0, 1., 0], [0, 0, 1], [1, 0, 0]]),
    'oblique': _rotation_z(20) @ np.diag([-1., -1, 1]),
}
SHAPE = (7, 9, 11)  # odd and all different, so a wrong transpose or flip cannot go unnoticed
ZOOMS = np.array([0.8, 0.9, 2.5])


def _affine(direction: str) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = DIRECTIONS[direction] * ZOOMS
    affine[:3, 3] = (-10., 20., -30.)
    return affine


def _voxels(dtype, shape=SHAPE) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    values = (values * 7919) % 4001 - 1500  # not monotonic in any axis
    if np.issubdtype(dtype, np.unsignedinteger):
        values = np.abs(values)
    if dtype == np.uint8:
        values = values % 117  # a label map
    if dtype == np.int32:
        values = values * 400_003 + 2 ** 29  # beyond float32's exact integers: the float32 rounding is part of the output
    out = values.astype(dtype)
    if np.issubdtype(dtype, np.floating):
        out = out / 3  # not representable exactly
        out.flat[5] = np.nan
    return out


# (stored dtype, scl_slope, scl_inter): CT (int16 + intercept), a non-integral scaling whose float64 arithmetic shows in
# the float32 result, unscaled integers (nibabel returns them unscaled), a label map, floats with NaN, large integers
SCALINGS = {
    'int16_ct': (np.int16, 1., -1024.),
    'int16_fraction': (np.int16, 0.37, 12.5),
    'uint16_small_slope': (np.uint16, 2.5e-3, 0.),
    'int16_unscaled': (np.int16, None, None),
    'uint8_labels': (np.uint8, None, None),
    'float32': (np.float32, None, None),
    'float64': (np.float64, None, None),
    'int32_large': (np.int32, None, None),
}


def _nifti(direction: str, scaling: str, shape=SHAPE) -> nibabel.Nifti1Image:
    """In memory: nibabel ignores scl_slope / scl_inter of images built from arrays, so the scaling is not set here."""
    return nibabel.Nifti1Image(_voxels(SCALINGS[scaling][0], shape), _affine(direction))


def _save_nifti(image: nibabel.Nifti1Image, fname: str, slope=None, inter=None) -> None:
    """Saves image's array as stored values with scl_slope / scl_inter (nibabel.save would choose its own scaling)."""
    nii = fname[:-3] if fname.endswith('.gz') else fname
    nibabel.save(image, nii)
    if slope is not None:
        with open(nii, 'r+b') as f:
            f.seek(112)  # scl_slope, scl_inter: float32 at bytes 112 and 116 of the NIfTI-1 header
            f.write(np.array([slope, inter], dtype=image.header.endianness + 'f4').tobytes())
    if nii != fname:
        with open(nii, 'rb') as src, gzip.open(fname, 'wb', compresslevel=1) as dst:
            shutil.copyfileobj(src, dst)
        os.remove(nii)


def _reference_nibabel(nib_images, reorient: bool):
    """NibabelIO(WithReorient).read_nibabel_images before the change: voxels, the affine(s) and spacing it returned."""
    images, original_affines, affines, spacings = [], [], [], []
    for nib_image in nib_images:
        original_affines.append(nib_image.affine)
        if reorient:
            nib_image = nib_image.as_reoriented(io_orientation(nib_image.affine))
        affines.append(nib_image.affine)
        spacings.append([float(i) for i in nib_image.header.get_zooms()[::-1]])
        images.append(nib_image.get_fdata().transpose((2, 1, 0))[None])
    return np.vstack(images, dtype=np.float32, casting='unsafe'), original_affines[0], affines[0], spacings[0]


def _reference_sitk(itk_images, orientation=None):
    """SimpleITKIO(WithReorient).read_sitk_images before the change: voxels and the (spacing, origin, direction)."""
    arrays, geometry = [], None
    for itk_image in itk_images:
        if orientation is not None:
            itk_image = sitk.DICOMOrient(itk_image, orientation)
        array = sitk.GetArrayFromImage(itk_image)
        arrays.append(array[None, None] if array.ndim == 2 else array[None] if array.ndim == 3 else array)
        geometry = geometry or (itk_image.GetSpacing(), itk_image.GetOrigin(), itk_image.GetDirection())
    return np.vstack(arrays, dtype=np.float32, casting='unsafe'), geometry


class TestNibabelReadersReturnTheSameVoxels(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _file(self, direction: str, scaling: str, ending: str = '.nii.gz') -> str:
        fname = os.path.join(self.tmp.name, f'{direction}_{scaling}{ending}')
        if not os.path.isfile(fname):
            _, slope, inter = SCALINGS[scaling]
            _save_nifti(_nifti(direction, scaling), fname, slope, inter)
            proxy = nibabel.load(fname).dataobj
            self.assertEqual((proxy.slope, proxy.inter),
                             (1., 0.) if slope is None else (np.float32(slope), np.float32(inter)))
        return fname

    def _assert_same(self, reader, make_images, reorient: bool):
        """make_images() must return fresh image objects: get_fdata() in the reference fills their caches."""
        data, properties = reader.read_nibabel_images(make_images())
        ref, ref_original_affine, ref_affine, ref_spacing = _reference_nibabel(make_images(), reorient)
        self.assertEqual(data.dtype, np.float32)
        self.assertTrue(data.flags.c_contiguous)
        self.assertEqual(data.shape, ref.shape)
        np.testing.assert_array_equal(data, ref)  # bit for bit, NaN where NaN
        np.testing.assert_array_equal(properties['nibabel_stuff']['original_affine'], ref_original_affine)
        if reorient:
            np.testing.assert_array_equal(properties['nibabel_stuff']['reoriented_affine'], ref_affine)
        self.assertEqual(properties['spacing'], ref_spacing)

    def _readers(self):
        return (NibabelIO(), False), (NibabelIOWithReorient(), True)

    def test_files(self):
        for (reader, reorient), direction, scaling, ending in (
                (r, d, s, e) for r in self._readers() for d in DIRECTIONS for s in SCALINGS for e in ('.nii.gz', '.nii')):
            with self.subTest(reader=type(reader).__name__, direction=direction, scaling=scaling, ending=ending):
                fname = self._file(direction, scaling, ending)
                self._assert_same(reader, lambda: [nibabel.load(fname)], reorient)
                data, _ = reader.read_images([fname])
                np.testing.assert_array_equal(data, reader.read_nibabel_images([nibabel.load(fname)])[0])

    def test_in_memory_images(self):
        """an image handed over with the file's proxy (as TotalSegmentator does) and images built from arrays"""
        for (reader, reorient), direction, scaling in (
                (r, d, s) for r in self._readers() for d in DIRECTIONS for s in SCALINGS):
            with self.subTest(reader=type(reader).__name__, direction=direction, scaling=scaling):
                fname = self._file(direction, scaling)

                def proxy():
                    image = nibabel.load(fname)
                    return [nibabel.Nifti1Image(image.dataobj, image.affine, image.header)]
                self._assert_same(reader, proxy, reorient)
                self._assert_same(reader, lambda: [_nifti(direction, scaling)], reorient)

    def test_conversion_in_several_blocks(self):
        for block_voxels in (1, 150, 10 ** 9):  # a slice per block, blocks with a shorter last one, one block
            with mock.patch.object(nibabel_reader_writer, '_BLOCK_VOXELS', block_voxels):
                for (reader, reorient), direction, scaling in (
                        (r, d, s) for r in self._readers() for d in ('RAS', 'LPS', 'SRA_permuted')
                        for s in ('int16_fraction', 'float32')):
                    with self.subTest(block_voxels=block_voxels, reader=type(reader).__name__, direction=direction,
                                      scaling=scaling):
                        fname = self._file(direction, scaling)
                        self._assert_same(reader, lambda: [nibabel.load(fname)], reorient)

    def test_several_channels(self):
        for reader, reorient in self._readers():
            for direction in ('RAS', 'LAS'):
                files = [self._file(direction, s) for s in ('int16_ct', 'float32', 'uint8_labels')]
                with self.subTest(reader=type(reader).__name__, direction=direction):
                    self._assert_same(reader, lambda: [nibabel.load(f) for f in files], reorient)
                    data, _ = reader.read_images(files)
                    self.assertEqual(data.shape[0], 3)

    def test_a_filled_cache_is_used_where_nibabel_uses_it(self):
        """get_fdata() returns a filled cache, edits included; as_reoriented reads the file. Both stay as they were."""
        for reader, reorient in self._readers():
            for direction in ('RAS', 'LPS'):
                with self.subTest(reader=type(reader).__name__, direction=direction):
                    image = nibabel.load(self._file(direction, 'int16_ct'))
                    image.get_fdata()[1, 2, 3] = 12345.
                    data, _ = reader.read_nibabel_images([image])
                    ref = _reference_nibabel([image], reorient)[0]
                    np.testing.assert_array_equal(data, ref)
                    edited = not (reorient and direction == 'LPS')
                    self.assertEqual(12345. in data, edited)

    def test_the_caller_image_holds_no_voxels_afterwards(self):
        for reader, _ in self._readers():
            with self.subTest(reader=type(reader).__name__):
                image = nibabel.load(self._file('LAS', 'int16_ct'))
                reader.read_nibabel_images([image])
                self.assertFalse(image.in_memory)

    def test_different_shapes_raise(self):
        other = os.path.join(self.tmp.name, 'other_shape.nii.gz')
        _save_nifti(_nifti('RAS', 'int16_ct', shape=(7, 9, 12)), other)
        for reader, _ in self._readers():
            with self.subTest(reader=type(reader).__name__):
                with self.assertRaises(RuntimeError):
                    reader.read_images([self._file('RAS', 'int16_ct'), other])


def _sitk_image(direction: str, pixel_dtype, shape_xyz=SHAPE) -> sitk.Image:
    image = sitk.GetImageFromArray(_voxels(pixel_dtype, shape_xyz[::-1]))
    image.SetSpacing(tuple(ZOOMS[:image.GetDimension()]))
    image.SetOrigin((-10., 20., -30.)[:image.GetDimension()])
    # SimpleITK's directions are LPS+: flip the first two rows of the RAS+ matrix
    image.SetDirection(tuple((np.diag([-1., -1, 1]) @ DIRECTIONS[direction]).ravel()))
    return image


SITK_DTYPES = (np.int16, np.uint8, np.uint16, np.int32, np.float32, np.float64)


class TestSimpleITKReadersReturnTheSameVoxels(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _file(self, image: sitk.Image, name: str) -> str:
        fname = os.path.join(self.tmp.name, name)
        sitk.WriteImage(image, fname)
        return fname

    def _assert_same(self, data, properties, ref, ref_geometry):
        self.assertEqual(data.dtype, np.float32)
        self.assertTrue(data.flags.c_contiguous)
        self.assertEqual(data.shape, ref.shape)
        np.testing.assert_array_equal(data, ref)
        self.assertEqual((properties['sitk_stuff']['spacing'], properties['sitk_stuff']['origin'],
                          properties['sitk_stuff']['direction']), ref_geometry)

    def _cases(self):
        for reader, orientation in ((SimpleITKIO(), None), (SimpleITKIOWithReorient(), 'RAS')):
            for direction in DIRECTIONS:
                for dtype in SITK_DTYPES:
                    yield reader, orientation, direction, dtype

    def test_files(self):
        for reader, orientation, direction, dtype in self._cases():
            with self.subTest(reader=type(reader).__name__, direction=direction, dtype=dtype.__name__):
                fname = self._file(_sitk_image(direction, dtype), f'{direction}_{dtype.__name__}.nii.gz')
                data, properties = reader.read_images([fname])
                self._assert_same(data, properties, *_reference_sitk([sitk.ReadImage(fname)], orientation))

    def test_in_memory_images(self):
        for reader, orientation, direction, dtype in self._cases():
            with self.subTest(reader=type(reader).__name__, direction=direction, dtype=dtype.__name__):
                image = _sitk_image(direction, dtype)
                data, properties = reader.read_sitk_images([image])
                self._assert_same(data, properties, *_reference_sitk([image], orientation))
                # the caller's image is untouched
                np.testing.assert_array_equal(sitk.GetArrayFromImage(image), _voxels(dtype, SHAPE[::-1]))

    def test_another_target_orientation(self):
        image = _sitk_image('oblique', np.int16)
        fname = self._file(image, 'oblique.nii.gz')
        for orientation in ('LPS', 'PIR'):
            with self.subTest(orientation=orientation):
                data, properties = SimpleITKIOWithReorient().read_images([fname], orientation)
                self._assert_same(data, properties, *_reference_sitk([sitk.ReadImage(fname)], orientation))
                data, properties = SimpleITKIOWithReorient().read_sitk_images([image], orientation=orientation)
                self._assert_same(data, properties, *_reference_sitk([image], orientation))

    def test_several_channels_and_2d(self):
        files = [self._file(_sitk_image('LAS', d), f'channel_{i}.nii.gz') for i, d in enumerate((np.int16, np.float32))]
        for reader, orientation in ((SimpleITKIO(), None), (SimpleITKIOWithReorient(), 'RAS')):
            with self.subTest(reader=type(reader).__name__):
                data, properties = reader.read_images(files)
                self.assertEqual(data.shape[0], 2)
                self._assert_same(data, properties, *_reference_sitk([sitk.ReadImage(f) for f in files], orientation))
        image_2d = sitk.GetImageFromArray(_voxels(np.float32, (9, 7)))
        fname = self._file(image_2d, 'image_2d.nrrd')
        data, properties = SimpleITKIO().read_images([fname])
        self._assert_same(data, properties, *_reference_sitk([sitk.ReadImage(fname)]))


# Reads one image in a fresh process and prints (peak RSS during the read - RSS before it, bytes of the result). The peak
# is the kernel's high-water mark of this process (VmHWM), reset right before the read.
_PEAK_SCRIPT = """
import sys
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO, SimpleITKIOWithReorient


def status_kib(field):
    return next(int(l.split()[1]) for l in open('/proc/self/status') if l.startswith(field + ':'))


reader = {c.__name__: c for c in (NibabelIO, NibabelIOWithReorient, SimpleITKIO, SimpleITKIOWithReorient)}[sys.argv[1]]()
with open('/proc/self/clear_refs', 'w') as f:
    f.write('5')  # resets VmHWM to the current RSS
before = status_kib('VmRSS')
data, _ = reader.read_images([sys.argv[2]])
print((status_kib('VmHWM') - before) * 1024, data.nbytes)
"""


@unittest.skipUnless(sys.platform.startswith('linux'), 'reads and resets the peak RSS through /proc (Linux)')
class TestReadersPeakMemory(unittest.TestCase):
    """
    A 256^3 image: the float32 result is 64 MiB. Before the change the readers peaked at 3-5 times that on top of the
    stored image; now at the stored image plus the result (nibabel: plus a few float64 blocks).
    """
    shape = (256, 256, 256)

    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        voxels = np.indices(cls.shape, dtype=np.int16).sum(0, dtype=np.int16)  # compresses well: keeps the test fast
        cls.nifti_int16, cls.nifti_float32 = {}, {}
        for direction in ('RAS', 'LAS'):
            # nibabel: compressed, so that it reads into memory instead of mapping the file
            cls.nifti_int16[direction] = os.path.join(cls.tmp.name, f'{direction}_int16.nii.gz')
            _save_nifti(nibabel.Nifti1Image(voxels, _affine(direction)), cls.nifti_int16[direction], 0.5, -1024.)
            image = sitk.GetImageFromArray(voxels.astype(np.float32))
            image.SetDirection(tuple((np.diag([-1., -1, 1]) @ DIRECTIONS[direction]).ravel()))
            cls.nifti_float32[direction] = os.path.join(cls.tmp.name, f'{direction}_float32.nii')
            sitk.WriteImage(image, cls.nifti_float32[direction])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _peak(self, reader: str, fname: str):
        result = subprocess.run([sys.executable, '-c', _PEAK_SCRIPT, reader, fname], capture_output=True, text=True,
                                check=True)
        peak, result_bytes = (int(i) for i in result.stdout.split())
        return peak, result_bytes

    def test_nibabel_readers(self):
        stored = int(np.prod(self.shape)) * 2
        for reader in ('NibabelIO', 'NibabelIOWithReorient'):
            for direction in ('RAS', 'LAS'):  # LAS: the reorienting reader flips an axis
                with self.subTest(reader=reader, direction=direction):
                    peak, result_bytes = self._peak(reader, self.nifti_int16[direction])
                    # the stored int16 image, the result and a few float64 blocks (old readers: 4x the result)
                    limit = stored + 1.25 * result_bytes
                    self.assertLess(peak, limit, f'{peak / 2 ** 20:.0f} MiB')

    def test_simpleitk_readers(self):
        for reader in ('SimpleITKIO', 'SimpleITKIOWithReorient'):
            for direction in ('RAS', 'LAS'):  # LAS: the reorienting reader makes a reoriented copy in ITK
                with self.subTest(reader=reader, direction=direction):
                    peak, result_bytes = self._peak(reader, self.nifti_float32[direction])
                    # the ITK image (float32 here, as large as the result) and the result (old readers: 3x and 4x)
                    self.assertLess(peak, 2.5 * result_bytes, f'{peak / 2 ** 20:.0f} MiB')


if __name__ == '__main__':
    unittest.main()
