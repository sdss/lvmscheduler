#!/usr/bin/env/python

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from astropy.time import Time

from lvmsurveysim.utils import wrapBlocking
from lvmsurveysim.schedule.scheduler import Atomic, Cals
from lvmsurveysim.exceptions import LVMSurveyOpsError
from lvmsurveysim.schedule.opsdb import OpsDB


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

@app.get("/next_tile")
async def next_tile(jd: float | None = None):
    """
    return the next tile
    """
    if jd:
        jd = jd
    else:
        now = Time.now()
        now.format = "jd"
        jd = now.value

    sched = await wrapBlocking(Atomic)

    await wrapBlocking(sched.prepare_for_night, np.floor(jd))

    try:
        tile_id, dither_pos = await wrapBlocking(sched.next_tile, jd)
        next_tile = {"tile_id": int(tile_id),
                     "jd": jd,
                     "dither_pos": dither_pos,
                     "errors": ""}
    except LVMSurveyOpsError:
        next_tile = {"tile_id": np.nan,
                     "jd": jd,
                     "dither_pos": 0,
                     "errors": "jd missing or invalid"}

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

    if tile_id is None and (ra is None or dec is None):
        raise HTTPException(status_code=418, detail="Must specify tile or RA/Dec")

    if not jd:
        now = Time.now()
        now.format = "jd"
        jd = now.value

    cals = await wrapBlocking(Cals, tile_id=tile_id, ra=ra, dec=dec, jd=jd)
    skies = await wrapBlocking(cals.choose_skies)
    standards = await wrapBlocking(cals.choose_standards)

    cal_dict = {"tile_id": tile_id,
                "sky_pks": [int(s) for s in skies],
                "standard_pks": [int(s) for s in standards]}

    return cal_dict


@app.put("/register_observation/")
async def register_observation(observation: Observation):
    """
    register a new observation
    """

    params = observation.dict()

    jd = params.get("jd")
    tile_id = params.get("tile_id")

    sched = await wrapBlocking(Atomic)
    await wrapBlocking(sched.prepare_for_night, np.floor(jd))

    obs_params = sched.scheduler.obs_info_helper(tile_id, jd)

    success = await wrapBlocking(OpsDB.add_observation, **params, **obs_params)

    return {"success": success}
