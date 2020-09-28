#!/usr/bin/env python

# Copyright 2018-2020 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import nbconvert
import skimage
import skimage.measure
import skimage.transform
import cv2
import warnings

with open("assignment8.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)


with open("assignment8.py", "w") as f:
    f.write(python_file)


from assignment8 import *


class TestSolution(unittest.TestCase):
    
    def test_kozeny_carmen_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.cla()
            plt.clf()
            kozeny_carmen_plot('poro_perm.dat')
            plt.savefig('poro_perm.png')

            gold_image = cv2.imread('poro_perm_gold.png')
            test_image = cv2.imread('poro_perm.png')

            test_image_resized = skimage.transform.resize(test_image, 
                                                          (gold_image.shape[0], gold_image.shape[1]), 
                                                          mode='constant')

            ssim = skimage.measure.compare_ssim(skimage.img_as_float(gold_image), test_image_resized, multichannel=True)
            assert ssim >= 0.5

    def test_contour_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.cla()
            plt.clf()
            contour_plot('Nechelik.dat')
            plt.savefig('Nechelik.png')

            gold_image = cv2.imread('Nechelik_gold.png')
            test_image = cv2.imread('Nechelik.png')

            test_image_resized = skimage.transform.resize(test_image, 
                                                          (gold_image.shape[0], gold_image.shape[1]), 
                                                          mode='constant')

            ssim = skimage.measure.compare_ssim(skimage.img_as_float(gold_image), test_image_resized, multichannel=True)
            assert ssim >= 0.5

if __name__ == '__main__':
        unittest.main()
