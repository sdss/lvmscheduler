#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#


# operations database and data classes for a survey tile and a survey observation
import pandas as pd
from astropy.table import Table

from lvmsurveysim.exceptions import LVMSurveyOpsError
import lvmsurveysim.utils.sqlite2astropy as s2a

from sdssdb.peewee.lvmdb import database

database.become_admin()

from sdssdb.peewee.lvmdb.lvmopsdb import Tile, Sky, Standard, Observation

# ########
# TODO: both of these methods will be using additional new tables
# ########


class OpsDB(object):
    """
    Interface the operations database for LVM. Makes the rest of the
    LVM Operations software agnostic to peewee or any other ORM we
    might be using one day.
    """
    def __init__(self):
        pass

    @classmethod
    def upload_tiledb(cls, tiledb=None, tile_table=None):
        """
        Saves a tile table to the operations database, optionally into a FITS table.
        The default is to update the tile database in SQL. No parameters are needed in 
        this case.
        Parameters
        ----------
        tiledb : `~lvmsurveysim.scheduler.TileDB`
            The instance of a tile database to save
        tile_table: astrop table to upload
        """

        assert tiledb or tile_table, "must pass something to upload"

        if not tile_table:
            tile_table = tiledb.tile_table
        s = s2a.astropy2peewee(tile_table, Tile, replace=True)
        return s

    @classmethod
    def load_tiledb(cls, version=None):
        """
        Load tile table, save version for later.

        TODO: connect to other tables to get completion, etc?
        """

        allRows = Tile.select().dicts()

        dataframe = pd.DataFrame(allRows)

        return Table.from_pandas(dataframe)

    @classmethod
    def load_sky(cls, ra=None, dec=None, radius=10, version=None):
        """
        Grab skies, optionally in a radius
        """

        allRows = Sky.select()
        if ra and dec:
            allRows = allRows.where(Sky.cone_search(ra, dec, radius))

        dataframe = pd.DataFrame(allRows.dicts())

        return Table.from_pandas(dataframe)

    @classmethod
    def load_standard(cls, ra=None, dec=None, radius=10, version=None):
        """
        Grab standards, optionally in a radius
        """

        allRows = Standard.select()
        if ra and dec:
            allRows = allRows.where(Standard.cone_search(ra, dec, radius))

        dataframe = pd.DataFrame(allRows.dicts())

        return Table.from_pandas(dataframe)

    @classmethod
    def retrieve_tile_ra_dec(cls, tile_id):
        tile = Tile.get(tile_id)

        return tile.ra, tile.dec

    @classmethod
    def load_history(cls, version=None):
        """
        Grab history
        """

        # table not implemented yet
        return None

    @classmethod
    def load_completion_status(cls, version=None):
        """
        Grab tile completion status
        """

        # table not implemented yet
        return None
