#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#


# operations database and data classes for a survey tile and a survey observation

from lvmsurveysim.exceptions import LVMSurveyOpsError

from sdssdb.lvmdb.lvmopsdb import Tile, Observation

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
        s = Tile.update({Tile.Status:status}).where(Tile.TileID==tileid).execute()
        if s==0:
            raise LVMSurveyOpsError('Attempt to set status on unknown TildID '+str(tileid))
        return s

    @classmethod
    def record_observation(cls, TileID, obstype, jd, lst, hz, obs_alt, lunation):
        '''
        Record an LVM Observation in the database.
        '''
        return Observation.insert(TileID=TileID, ObsType=obstype,
                                  JD=jd, LST=lst, Hz=hz, Alt=obs_alt, Lunation=lunation).execute()
