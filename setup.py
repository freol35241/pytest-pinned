#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()


setup(
    name='pytest-pinpoint',
    version='0.1.0',
    author='Fredrik Olsson',
    author_email='freol@outlook.com',
    maintainer='Fredrik Olsson',
    maintainer_email='freol@outlook.com',
    license='MIT',
    url='https://github.com/freol35241/pytest-pinpoint',
    description='A simple pytest plugin for pinning tests',
    long_description=read('README.md'),
    py_modules=['pytest_pinpoint'],
    python_requires='>=3.5',
    install_requires=['pytest>=3.5.0'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: Pytest',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points={
        'pytest11': [
            'pinpoint = pytest_pinpoint',
        ],
    },
)
