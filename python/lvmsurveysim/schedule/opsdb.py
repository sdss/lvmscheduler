#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#


# operations database and data classes for a survey tile and a survey observation

from lvmsurveysim.exceptions import LVMSurveyOpsError
import lvmsurveysim.utils.sqlite2astropy as s2a

from sdssdb.peewee.lvmdb import database

database.become_admin()

from sdssdb.peewee.lvmdb.lvmopsdb import Tile, Observation

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
    def update_tile_status(cls, tileid, status):
        '''
        Update the tile Status column in the tile database.
        '''
        s = Tile.update({Tile.Status: status}).where(Tile.TileID == tileid).execute()
        if s == 0:
            raise LVMSurveyOpsError('Attempt to set status on unknown TildID '+str(tileid))
        return s

    @classmethod
    def record_observation(cls, TileID, obstype, jd, lst, hz, obs_alt, lunation):
        '''
        Record an LVM Observation in the database.
        '''
        return Observation.insert(TileID=TileID, JD=jd, LST=lst, Hz=hz,
                                  Alt=obs_alt, Lunation=lunation).execute()

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
