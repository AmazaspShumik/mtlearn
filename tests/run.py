import unittest


def run_tests():
    loader = unittest.TestLoader()
    tests = loader.discover('.', pattern='*_test.py')
    test_runner = unittest.TextTestRunner()
    test_runner.run(tests)


if __name__ == '__main__':
    run_tests()