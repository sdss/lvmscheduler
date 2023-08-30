#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#


# Very quick hack to load an sqlite3 table using peewee into as astropy.Table object
# and save an astropy.Table into an sqlite3 database using a matching peewee model

import astropy.table
import numpy
from peewee import chunked

__all__ = ['peewee2astropy', 'astropy2peewee']


def peewee2astropy(model):
    '''
    Create an `astropy.table.Table` object using a peewee database
    model class. Data are read from the corresponding SQL database.

    Parameters
    ----------
    model : ~peewee.Model
        The model corresponding to the table to be read

    Returns
    -------
    table : ~astropy.table.Table 
        astropy Table object with the data from the database table. Column
        names match the attribute names in the model.
    '''
    cols = model._meta.columns
    cnames = [c for c in cols]

    cursor = model._meta.database.execute_sql('SELECT * from '+model._meta.table_name)
    results = cursor.fetchall()

    results = numpy.rec.fromrecords(list(results), names=cnames)

    table = astropy.table.Table()
    for _, column in enumerate(results.dtype.names):
        table.add_column(results[column], name=column)

    return table


def astropy2peewee(table, model):
    '''
    Save an `astropy.table.Table` into a pewwee SQL database using a
    given model. The model's attributes MUST match the table's column by name and type.
    Optionally insert or replace rows in the database.

    Parameters
    ----------
    table : ~astropy.table.Table
        astropy Table object with the data to be stored. Columns
        must match the attribute names in the model exactly.

    model : ~peewee.Model
        The peewee model of the database table to store into
    '''
    cols = model._meta.columns
    dbc = [c for c in cols]
    tbc = table.colnames
    for d, t in zip(dbc, tbc):
        assert d == t, "column mismatch"
    # this set stuff isn't working, even with equality of all elements and order
    # assert (len(dbc) == len(tbc)) & (set(dbc) == set(tbc)), "sets of columns do not match."
    db = model._meta.database
    with db.atomic():
        s = model.insert_many(table).execute()
    return s
