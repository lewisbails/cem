#!/usr/bin/env python

"""The setup script."""
import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.rst").read_text()

requirements = [
    'pandas',
    'numpy',
]

setup(
    author="Lewis Bails",
    author_email='lewis.bails@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="A Python implmentation of Coarsened Exact Matching for causal inference",
    install_requires=requirements,
    license="MIT license",
    long_description=README,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords='cem',
    name='cem',
    packages=find_packages(include=['cem', 'cem.*']),
    url='https://github.com/lewisbails/cem',
    version='0.1.5',
    zip_safe=False,
)
