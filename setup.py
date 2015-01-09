#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "gpyk_wrapper",
    version = "0.1",
    author = "Daniel Beck",
    packages = ["gpyk_wrapper"],
    url = "https://github.com/beckdaniel/gpyk_wrapper",
    long_description = read('README.md'),
)
