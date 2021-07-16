import os
from setuptools import setup, find_packages

packages = [x for x in find_packages('.') if x.startswith('carriage_return')]

setup(
    name = "carriage_return",
    packages=packages,
)


