"""Setup script for mtlearn"""

from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup

import mtlearn
import mtlearn.experiments

setup(
    name='mtlearn',
    version='0.1.0a1',
    description="multi-task learning package with reproduced papers",
    url='https://github.com/AmazaspShumik/mtlearn',
    author='Amazasp Shaumyan',
    author_email='amazasp.shaumyan@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=0.15.1',
        'scikit-learn>=0.21',
        'tensorflow>=2.0.0-beta1',
        'keras-tuner>=1.0.0'],
    test_suite='tests',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'],
)
