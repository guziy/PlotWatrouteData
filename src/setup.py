__author__="huziy"
__date__ ="$21 sept. 2010 19:07:30$"

from setuptools import setup,find_packages

setup (
  name = 'PlotWatrouteData',
  version = '0.1',
  packages = find_packages(),

  # Declare your packages' dependencies here, for eg:
  install_requires=['', "numpy", "matplotlib", "netCDF4", "scipy"],

  # Fill in these to make your Egg ready for upload to
  # PyPI
  author = 'huziy',
  author_email = '',

  summary = 'Just another Python package for the cheese shop',
  url = '',
  license = '',
  long_description= 'Long description of the package',

  # could also include long_description, download_url, classifiers, etc.

  
)