# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-09-27 11:08:07
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last Modified time: 2017-09-27 11:20:46

from __future__ import print_function, division, absolute_import
import os
from invoke import Collection, task


# This file contains tasks that can be easily run from the shell terminal using
# the Invoke python package
# If you do not have invoke, install it with pip install
# To list the tasks available, type invoke --list from the top-level repo directory

@task
def clean_docs(ctx):
    ''' Cleans up the Sphinx docs '''
    print('Cleaning the docs')
    ctx.run("rm -rf docs/sphinx/_build", pty=True)


@task(clean_docs)
def build_docs(ctx):
    ''' Builds the Sphinx docs '''
    print('Building the docs')
    os.chdir('docs/sphinx')
    ctx.run("make html", pty=True)


@task
def show_docs(ctx):
    """Shows the Sphinx docs"""
    print('Showing the docs')
    os.chdir('docs/sphinx/_build/html')
    ctx.run('open ./index.html', pty=True)


@task
def clean(ctx):
    ''' Cleans up the crap before a Pip build '''
    print('Cleaning')
    ctx.run("rm -rf htmlcov", pty=True)
    ctx.run("rm -rf build", pty=True)
    ctx.run("rm -rf dist", pty=True)
    ctx.run("rm -rf lvmsurveysim.egg-info", pty=True)


@task(clean)
def deploy(ctx):
    ''' Deploy the project to pypi '''
    print('Deploying to Pypi!')
    ctx.run("python setup.py sdist bdist_wheel --universal", pty=True)
    ctx.run("twine upload dist/*", pty=True)


# create a collection of tasks
ns = Collection(clean, deploy)

# create a sub-collection for the doc tasks
docs = Collection('docs')
docs.add_task(build_docs, 'build')
docs.add_task(clean_docs, 'clean')
docs.add_task(show_docs, 'show')
ns.add_collection(docs)
