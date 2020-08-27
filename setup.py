# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:51:01 2020

@author: user
"""
from setuptools import setup

setup(
      name="typhoonify",
      version="0.1",
      description="various utils for generating typhoon wind field",
      packages=["typhoonify"],
      install_requires=["scipy"]
      )