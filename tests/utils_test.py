from unittest import TestCase

from mtlearn.utils import has_arg


class TestUtils(TestCase):
    """ Test utils """

    def test_has_arg(self):

        def test_func_diff(a, b):
            return a-b

        def test_func_sum(a=2, b=3):
            return a+b

        self.assertTrue(has_arg(test_func_diff,"a"))
        self.assertTrue(has_arg(test_func_sum,"b"))
        self.assertFalse(has_arg(test_func_sum,"c"))
        self.assertFalse(has_arg(test_func_diff,"z"))