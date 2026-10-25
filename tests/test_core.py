import unittest
from src.core import PATTERN,normalize,correct_by_mask,plate_type
import numpy as np
class CoreTest(unittest.TestCase):
 def test_valid(self):
  self.assertIsNotNone(PATTERN.fullmatch('A123BC777'))
  self.assertIsNotNone(PATTERN.fullmatch('K555OP25'))
  self.assertIsNone(PATTERN.fullmatch('A123BC9999'))
 def test_normalize(self):self.assertEqual(normalize('А123ВС77'),'A123BC77')
 def test_mask(self):self.assertEqual(correct_by_mask('A12OBC77'),'A120BC77')
 def test_yellow(self):
  im=np.zeros((80,500,3),dtype=np.uint8); im[:]=(20,215,250)
  self.assertEqual(plate_type(im),'type1b')
if __name__=='__main__':unittest.main()
