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
import unittest
import unittest2

import numpy as np
from nnunet.network_architecture.neural_network import SegmentationNetwork


class TestSlidingWindow(unittest2.TestCase):
    def setUp(self) -> None:
        pass

    def _verify_steps(self, steps, patch_size, image_size, step_size):
        debug_information = 'steps= %s\nimage_size= %s\npatch_size= %s\nstep_size= %0.4f' % (str(steps),
                                                                                             str(image_size),
                                                                                             str(patch_size), step_size)
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        # this code is copied form the current implementation. Not ideal, but I don't know hoe else to the the
        # expected num_steps
        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels,
                                                                      patch_size)]

        self.assertTrue(all([len(i) == num_steps[j] for j, i in enumerate(steps)]),
                        'steps do not match expected num_steps %s. \nDebug: %s' % (str(num_steps), debug_information))

        for dim in range(len(steps)):
            # first step must start at 0
            self.assertTrue(steps[dim][0] == 0)

            # last step + patch size must equal to image size
            self.assertTrue(steps[dim][-1] + patch_size[dim] == image_size[dim], 'not the whole image is covered. '
                                                                                 '\nDebug: %s' % debug_information)

            # there cannot be gaps between adjacent predictions
            self.assertTrue(all([steps[dim][i + 1] <= steps[dim][i] + patch_size[dim] for i in
                                 range(num_steps[dim] - 1)]), 'steps are not overlapping or touching. dim: %d, steps:'
                                                              ' %s, image_size: %s, patch_size: %s, step_size: '
                                                              '%0.4f' % (
                                dim, str(steps[dim]), str(image_size[dim]), str(patch_size[dim]), step_size))

            # two successive steps cannot be further apart than target_step_sizes_in_voxels
            self.assertTrue(all([steps[dim][i] + np.ceil(target_step_sizes_in_voxels[dim]) >= steps[dim][i + 1] for i
                                 in range(num_steps[dim] -1)]),
                            'consecutive steps are too far apart. Steps: %s, dim: %d. \nDebug: %s' %
                            (str(steps[dim]), dim, debug_information))

    def test_same_image_and_patch_size_3d(self):
        image_size = (24, 845, 321)
        patch_size = (24, 845, 321)

        # this should always return steps=[[0],[0],[0]] no matter what step_size we choose
        expected_result = [[0], [0], [0]]
        step_size = 1
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == expected_result)

        step_size = 0.125
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == expected_result)

        step_size = 0.5
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == expected_result)

    def test_same_image_and_patch_size_2d(self):
        image_size = (123, 143)
        patch_size = (123, 143)

        # this should always return steps=[[0],[0]] no matter what step_size we choose
        expected_result = [[0], [0]]
        step_size = 1
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == expected_result)

        step_size = 0.125
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == expected_result)

        step_size = 0.5
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == expected_result)

    def test_some_manually_verified_combinations(self):
        image_size = (128, 260)
        patch_size = (64, 130)
        step_size = 0.5

        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0, 32, 64], [0, 65, 130]])

        step_size = 0.85
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0, 32, 64], [0, 65, 130]])

        step_size = 1
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0, 64], [0, 130]])

        # an example from task02
        image_size = (146, 176, 148)
        patch_size = (128, 128, 128)
        step_size = 0.5

        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0, 18], [0, 48], [0, 20]])

        # heart
        image_size = (130, 320, 244)
        patch_size = (80, 192, 160)
        step_size = 0.5

        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0, 25, 50], [0, 64, 128], [0, 42, 84]])

        step_size = 0.75
        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0, 50], [0, 128], [0, 84]])

        # liver
        image_size = (424, 456, 456)
        patch_size = (128, 128, 128)
        step_size = 0.5

        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0, 59, 118, 178, 237, 296],
                                  [0, 55, 109, 164, 219, 273, 328],
                                  [0, 55, 109, 164, 219, 273, 328]]
                        )

        # hippo
        image_size = (40, 56, 40)
        patch_size = (40, 56, 40)
        step_size = 0.5

        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0],
                                  [0],
                                  [0]]
                        )

        # hepaticvessel
        image_size = (94, 308, 308)
        patch_size = (64, 192, 192)
        step_size = 0.5

        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
        self.assertTrue(steps == [[0, 30],
                                  [0, 58, 116],
                                  [0, 58, 116]]
                        )

    def test_loads_of_combinations(self):
        """
        We now take a large number of random combinations and perform sanity checks
        :return:
        """
        for _ in range(5000):
            dim = np.random.choice((2, 3))

            patch_size = tuple(np.random.randint(16, 1024, dim))
            image_size = tuple(np.random.randint(i / 2, i * 10) for i in patch_size)
            image_size = tuple(max(image_size[i], patch_size[i]) for i in range(len(image_size)))
            step_size = np.random.uniform(0.01, 1)

            #print(image_size, patch_size, step_size)

            steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, image_size, step_size)
            self._verify_steps(steps, patch_size, image_size, step_size)


if __name__ == '__main__':
    unittest.main()
