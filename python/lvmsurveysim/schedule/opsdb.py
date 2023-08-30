#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: John Donor (j.donor@tcu.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#


# operations database and data classes for a survey tile and a survey observation
import os
import pandas as pd
from astropy.table import Table
from astropy.time import Time 
from peewee import fn

from lvmsurveysim import __version__ as schedVer
import lvmsurveysim.utils.sqlite2astropy as s2a

from sdssdb.peewee.lvmdb import database

database.become_admin()

from sdssdb.peewee.lvmdb.lvmopsdb import (Tile, Sky, Standard, Observation,
                                          CompletionStatus, Dither, Exposure,
                                          ExposureFlavor, ObservationToStandard,
                                          ObservationToSky, Weather,
                                          Version)


class OpsDB(object):
    """
    Interface the operations database for LVM. Makes the rest of the
    LVM Operations software agnostic to peewee or any other ORM we
    might be using one day.
    """
    def __init__(self):
        pass

    @classmethod
    def upload_tiledb(cls, tiledb=None, tile_table=None,
                      version="forgot"):
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

        dbVer, new = Version.get_or_create(label=version,
                                           sched_tag=schedVer)
        
        tile_table["version_pk"] = [int(dbVer.pk) for t in tile_table["tile_id"]]

        s = s2a.astropy2peewee(tile_table, Tile)
        return s

    @classmethod
    def load_tiledb(cls, version=None):
        """
        Load tile table, save version for later.

        version : int or str
            pk or label for tiling version
        """

        if type(version) == int:
            ver = Version.get(pk=version)
        elif type(version) == str:
            ver = Version.get(label=version)
        else:
            ver = Version.get(label=os.getenv("TILE_VER"))

        allRows = Tile.select()\
                      .where(Tile.version_pk == ver.pk)\
                      .order_by(Tile.tile_id.asc())

        dataframe = pd.DataFrame(allRows.dicts())

        return Table.from_pandas(dataframe)

    @classmethod
    def load_sky(cls, ra=None, dec=None, radius=None, version=None):
        """
        Grab skies, optionally in a radius
        """

        allRows = Sky.select()
        if ra and dec and radius:
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
    def retrieve_tile_dithers(cls, tile_id):
        pos = Tile.select(Dither.position)\
                  .join(Dither)\
                  .join(Observation)\
                  .switch(Dither)\
                  .join(CompletionStatus)\
                  .where(CompletionStatus.done,
                         Tile.tile_id == tile_id)

        return [p.position for p in pos]

    @classmethod
    def load_history(cls, version=None, tile_ids=None):
        """
        Grab history

        version : TBD, if needed

        tile_ids: iterable of ints, hist will be returned in this order
        """

        if tile_ids is None:
            tq = Tile.select(Tile.tile_id).order_by(Tile.tile_id.asc())
            tile_ids = [t.tile_id for t in tq]

        hist = Tile.select(Tile.tile_id,
                           fn.Count(Dither.pk).alias("count"))\
                   .join(Dither)\
                   .join(CompletionStatus)\
                   .where(CompletionStatus.done)\
                   .group_by(Tile.tile_id).dicts()

        hist = {h["tile_id"]: h["count"] * 900 for h in hist}

        total_time = [hist.get(t, 0) for t in tile_ids]
        return total_time

    @classmethod
    def load_completion_status(cls, version=None):
        """
        Possibly unreasonable if we're doing completion
        based on dithers, calculate elsewhere?

        Grab tile completion status
        """

        # table not implemented yet
        return None

    @classmethod
    def add_observation(cls, tile_id=None, exposure_no=None, dither=0, jd=0.0,
                        seeing=10.0, standards=[], skies=[],
                        lst=None, hz=None, alt=None, lunation=None):
        """
        add a recent observation to the DB
        """

        if exposure_no is None:
            return False

        if tile_id is None:
            obs = None
        else:
            dither_pos, created = Dither.get_or_create(tile_id=tile_id, position=dither)
            dither_stat, created = CompletionStatus.get_or_create(dither=dither_pos)
            dither_stat.update(done=True, by_pipeline=False).execute()

            obs = Observation.create(dither=dither_pos,
                                     jd=jd,
                                     lst=lst,
                                     hz=hz,
                                     alt=alt,
                                     lunation=lunation)
            
            Weather.create(obs_id=obs.obs_id, seeing=seeing)

        sciece_flavor = ExposureFlavor.get(label="Science")

        start_time = Time(jd, format="jd").datetime

        Exposure.create(observation=obs,
                        exposure_no=exposure_no,
                        exposure_flavor=sciece_flavor,
                        start_time=start_time,
                        exposure_time=900)

        if obs is None:
            return True

        standard_dicts = list()
        for s in standards:
            standard_dicts.append({"standard_pk": s,
                                   "obs_id": obs.obs_id})
        
        res = ObservationToStandard.insert_many(standard_dicts).execute()

        sky_dicts = list()
        for s in skies:
            sky_dicts.append({"sky_pk": s,
                              "obs_id": obs.obs_id})
        
        res = ObservationToSky.insert_many(sky_dicts).execute()

        return True

    @classmethod
    def tile_info(cls, tile_id):
        """
        return info on a single tile_id
        """

        tile = Tile.get(tile_id=tile_id)

        return tile.__data__
