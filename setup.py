# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:51:01 2020

@author: user
"""
from setuptools import setup

setup(
      name="typhoon_model",
      version="0.3",
      description="various utils for generating typhoon wind field",
      packages=["typhoon_model"],
      install_requires=["scipy, numpy, netcdf4, pandas"]
      )
