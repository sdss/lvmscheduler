lvmsurveysim
============

Survey tiling, operations, and simulations for LVM.

|travis| |coveralls| |docs| |py36|

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://sdss-lvmsurveysim.readthedocs.io/en/latest/?badge=latest

.. |py36| image:: https://img.shields.io/badge/python-3.6-blue.svg

.. |travis| image:: https://travis-ci.org/sdss/lvmsurveysim.svg?branch=master
    :target: https://travis-ci.org/sdss/lvmsurveysim

.. |coveralls| image:: https://coveralls.io/repos/github/sdss/lvmsurveysim/badge.svg?service=github
    :target: https://coveralls.io/github/sdss/lvmsurveysim

Now ready for simulations (again)
---------------------------------

``$ git clone https://github.com/sdss/lvmscheduler.git``

``$ cd lvmscheduler``
    
``$ pip install -e .``
    
``$ lvm_sim -w tiledb -o figsDir``

``-w`` for "write" a tiledb so you don't have to tile every time

``-r`` to "read" a written tiledb

``-o`` for "out"; a directory to save the output figures
