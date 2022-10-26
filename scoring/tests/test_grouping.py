
# Home Credit Python Scoring Library and Workflow
# Copyright © 2017-2020, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller,
# Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
# Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from pandas.testing import assert_frame_equal
import shutil, tempfile
from os import path
from scoring.grouping import woe, tree_based_grouping, auto_group_categorical, Grouping
import pandas as pd


class TestWoe(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(woe([], [0, 1, 0 ,1], 0), 0)

    def test_positive(self):
        self.assertTrue(woe([0, 0, 1], [0 , 0 ,0 ,1, 1, 1], 0) > 0.)
    def test_negative(self):
        self.assertTrue(woe([1, 1, 0], [0 , 0 ,0 ,1, 1, 1], 0) < 0.)
      
class TestGrouping(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    
    def test_tree_based_grouping(self):
        x = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        bins = tree_based_grouping(x, y, group_count = 3, min_samples = 1)
        assert_array_equal(bins, np.array([-np.inf, 0.5, 1.5, np.inf]))
    
    def test_grouping(self):
        X = pd.DataFrame({'a': [0, 0, 1, 1]})
        y = [0, 0, 1, 1]
        grouping = Grouping(columns = ['a'])
        grouping.fit(X, y)
        assert_almost_equal(grouping.bins_data_['a']['bins'], [-np.inf, 0.5, np.inf])
        assert_almost_equal(grouping.bins_data_['a']['woes'], [6.90875478, -6.90875478])
        assert_almost_equal(grouping.bins_data_['a']['nan_woe'], 0.)
        X_woe = grouping.transform(X)
        assert_frame_equal(X_woe, pd.DataFrame({'a': [6.908755, 6.908755, -6.908755, -6.908755]}))
        

    def test_grouping2(self):
        X = pd.DataFrame({'a': [0, 0, np.nan, np.nan, 1]})
        y = [0, 0, 0, 1, 1]
        grouping = Grouping(columns = ['a'])
        grouping.fit(X, y)
        assert_almost_equal(grouping.bins_data_['a']['bins'], [-np.inf, 0.5, np.inf])
        assert_almost_equal(grouping.bins_data_['a']['woes'], [6.90875478, -6.90875478])
        assert_almost_equal(grouping.bins_data_['a']['nan_woe'], 0)
        X_woe = grouping.transform(X)
        assert_frame_equal(X_woe, pd.DataFrame({'a': [6.908755, 6.908755, 0, 0, -6.908755]}))
        
    def test_grouping3(self):
        X = pd.DataFrame({'a': [0, 0, np.nan, np.nan, 1]})
        y = [0, 0, 0, 1, 1]
        grouping = Grouping(columns = ['a'])
        grouping.fit(X, y)
        filename = path.join(self.test_dir, 'grouping')
        grouping.save(filename)
        X_woe_1 = grouping.transform(X)
        grouping.load(filename)
        X_woe_2 = grouping.transform(X)
        assert_frame_equal(X_woe_1, X_woe_2)

    #def test_nan(self):
    #    tree_based_grouping([1,1, np.nan], [0,1, 1], 2, 1)
    
    def test_cat(self):
        x = ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', np.nan]
        y = [0,   0,   0,   1,   1,   0,   1,   0,   1,   0]
        
        bins, woes = auto_group_categorical(x, y, group_count = 2, min_samples = 1, min_cat_samples = 3, woe_smooth_coef=0.01)
        print(bins, woes)
        
    def test_cat1(self):
        x = ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'c', 'c']
        y = [0,   0,   0,   1,   1,   0,   1,   0,   1,   0]
        
        bins, woes = auto_group_categorical(x, y, group_count = 2, min_samples = 1, min_cat_samples = 3, woe_smooth_coef=0.01)
        print(bins, woes)

    def test_all(self):
        x = ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'c', 'c']
        x1 = [1., 1, 1, 2, 3, 4, 5, 6, 7, 8]
        y = [0,   0,   0,   1,   1,   0,   1,   0,   1,   0]
        
        grouping = Grouping(['x1'], ['x'], group_count = 2, min_samples = 1, min_cat_samples = 3, woe_smooth_coef=0.01)
        X = pd.DataFrame()
        X['x'] = x
        X['x1'] = x1
        grouping.fit(X, y)
        print(grouping.transform(X))
        #print(grouping.bins_data_)
        grouping.save('tst')
        grouping.load('tst')
        print(grouping.transform(X))
        
      
if __name__ == '__main__':
    unittest.main()
