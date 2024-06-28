#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('VERSION', 'r') as _f:
    __version__ = _f.read().strip()

setup(
    name='pyacal',
    version=__version__,
    author='python-accelerator-middle-layer',
    description='Accelerator Middle Layer for Python',
    url='https://github.com/python-accelerator-middle-layer/pyacal',
    download_url='https://github.com/python-accelerator-middle-layer/pyacal',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=find_packages(),
    package_data={'pyacal': ['VERSION', ]},
    zip_safe=False
)
