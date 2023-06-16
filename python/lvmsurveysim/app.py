#!/usr/bin/env/python
from quart import Quart, jsonify, request
import numpy as np
from astropy.time import Time

from lvmsurveysim.utils import wrapBlocking
from lvmsurveysim.schedule.scheduler import Atomic, Cals
from lvmsurveysim.exceptions import LVMSurveyOpsError

app = Quart(__name__)


@app.route("/next_tile", methods=["GET"])
async def next_tile():
    """
    return the next tile
    """
    if "jd" in request.args:
        jd = float(request.args["jd"])
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

    return jsonify(next_tile)


@app.route("/cals", methods=["GET"])
async def cals():
    """
    return the next tile
    """
    tile_id = None
    ra = None
    dec = None
    if "tile_id" in request.args:
        tile_id = int(request.args["tile_id"])
    if "ra" in request.args and "dec" in request.args:
        ra = float(request.args["ra"])
        dec = float(request.args["dec"])

    if "jd" in request.args:
        jd = float(request.args["jd"])
    else:
        now = Time.now()
        now.format = "jd"
        jd = now.value

    cals = await wrapBlocking(Cals, tile_id=tile_id, ra=ra, dec=dec, jd=jd)
    skies = await wrapBlocking(cals.choose_skies)
    standards = await wrapBlocking(cals.choose_standards)

    cal_dict = {"tile_id": tile_id,
                "sky_pks": [int(s) for s in skies],
                "standard_pks": [int(s) for s in standards]}

    return jsonify(cal_dict)
