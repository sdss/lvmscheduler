#!/usr/bin/env/python

import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from astropy.time import Time

from lvmsurveysim.utils import wrapBlocking
from lvmsurveysim.schedule.scheduler import Atomic, Cals
from lvmsurveysim.exceptions import LVMSurveyOpsError
from lvmsurveysim.schedule.opsdb import OpsDB

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
LOGFILE = "/data/logs/lvmscheduler/current.log"
fh = RotatingFileHandler(LOGFILE, maxBytes=(1048576 * 5), backupCount=7)
fh.setFormatter(format)
logger.addHandler(fh)


class Observation(BaseModel):
    dither: int
    tile_id: int
    jd: float
    seeing: float
    standards: list
    skies: list
    exposure_no: int


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "[Obi Wan]: This is not the page you're looking for"}


@app.get("/tile_info")
async def tile_info(tile_id: int):
    """
    return the next tile
    """

    logger.info(f"info for tile_id {tile_id} requested")

    info = OpsDB.tile_info(tile_id)

    return info


@app.get("/next_tile")
async def next_tile(jd: float | None = None):
    """
    return the next tile
    """

    logger.info(f"tile request for JD {jd}")
    if jd:
        jd = jd
    else:
        now = Time.now()
        now.format = "jd"
        jd = now.value

    sched = await wrapBlocking(Atomic)

    await wrapBlocking(sched.prepare_for_night, np.floor(jd))

    errors = []

    if jd < sched.scheduler.evening_twi:
        new_jd = sched.scheduler.evening_twi
        logger.info(f"jd {jd} < evening twilight, setting to {new_jd}")
        errors.append("JD too early, using evening twilight")
        jd = new_jd
    elif jd > sched.scheduler.morning_twi:
        new_jd = sched.scheduler.morning_twi - 1 / 24
        logger.info(f"jd {jd} > morning twilight, setting to {new_jd}")
        errors.append("JD too late, using morning twilight - 1 hr")
        jd = new_jd

    logger.info(f"pulling tile for JD {jd}")

    try:
        tile_id, dither_pos, pos = await wrapBlocking(sched.next_tile, jd)
        next_tile = {"tile_id": int(tile_id),
                     "jd": jd,
                     "dither_pos": dither_pos,
                     "tile_pos": pos,
                     "errors": errors,
                     "coord_order": ["ra", "dec", "pa"]}
    except LVMSurveyOpsError as E:
        logger.warning(f"caught exception {E}")

        raise HTTPException(status_code=418, detail="There is a problem with the JD")

    logger.info(f"tile {tile_id} with dither {dither_pos} JD {jd}")

    return next_tile


@app.get("/cals")
async def cals(
    tile_id: int | None = None,
    ra: float | None = None,
    dec: float | None = None,
    jd: float | None = None
):
    """
    return cals for tile or location
    """

    logger.info(f"cals requested for tile_id: {tile_id}, ra {ra}, dec {dec}, JD {jd}")

    if tile_id is None and (ra is None or dec is None):
        raise HTTPException(status_code=418, detail="Must specify tile or RA/Dec")

    if not jd:
        now = Time.now()
        now.format = "jd"
        jd = now.value

    cals = await wrapBlocking(Cals, tile_id=tile_id, ra=ra, dec=dec, jd=jd)
    skies, sky_pos = await wrapBlocking(cals.choose_skies)
    standards, stan_pos = await wrapBlocking(cals.choose_standards)

    logger.info("standards: " + ",".join([str(s) for s in standards]))
    logger.info("skies: " + ",".join([str(s) for s in skies]))

    cal_dict = {"tile_id": tile_id,
                "sky_pks": [int(s) for s in skies],
                "standard_pks": [int(s) for s in standards],
                "sky_pos": sky_pos,
                "standard_pos": stan_pos,
                "coord_order": ["ra", "dec"]}

    return cal_dict


@app.put("/register_observation/")
async def register_observation(observation: Observation):
    """
    register a new observation
    """

    params = observation.dict()

    logger.info("attempting to register observation")
    logger.info(params)

    jd = params.get("jd")
    tile_id = params.get("tile_id")

    sched = await wrapBlocking(Atomic)
    await wrapBlocking(sched.prepare_for_night, np.floor(jd))

    obs_params = sched.scheduler.obs_info_helper(tile_id, jd)

    success = await wrapBlocking(OpsDB.add_observation, **params, **obs_params)

    logger.info(f"register observation reported {success}")

    return {"success": success}
