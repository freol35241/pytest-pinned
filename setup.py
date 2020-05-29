#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()

# Parse the requirements-txt file and use for install_requires in pip
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pytest-pinned',
    version='0.2.0',
    author='Fredrik Olsson',
    author_email='freol@outlook.com',
    maintainer='Fredrik Olsson',
    maintainer_email='freol@outlook.com',
    license='MIT',
    url='https://github.com/freol35241/pytest-pinned',
    description='A simple pytest plugin for pinning tests',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    py_modules=['pytest_pinned'],
    python_requires='>=3.5',
    install_requires=required,
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
            'pinned = pytest_pinned',
        ],
    },
)
