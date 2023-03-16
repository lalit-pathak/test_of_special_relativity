#!/usr/bin/env python
"""
setup.py file for TaylorF2_Mod pycbc waveform plugin package
"""

from setuptools import Extension, setup, Command
from setuptools import find_packages

VERSION = '0.0.dev0'

setup (
    name = 'pycbc_TaylorF2_mod_full',
    version = VERSION,
    description = 'Modified TaylorF2 waveform plugin for PyCBC',
    long_description = open('descr.rst').read(),
    author = 'Lalit Pathak',
    author_email = 'lalit.pathak@iitgn.ac.in',
    url = 'http://www.pycbc.org/',
    download_url = 'https://github.com/gwastro/TaylorF2_mod_full/tarball/v%s' % VERSION,
    keywords = ['pycbc', 'signal processing', 'gravitational waves'],
    install_requires = ['pycbc'],
    py_modules = ['TaylorF2_mod_full'],
    entry_points = {"pycbc.waveform.fd":"TaylorF2_full = TaylorF2_mod_full:TaylorF2_full"}, 
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)