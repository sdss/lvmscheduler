#!/usr/bin/env/python
from quart import Quart, jsonify, request
import numpy as np

from lvmsurveysim.utils import wrapBlocking
from lvmsurveysim.schedule.scheduler import Atomic, Cals

app = Quart(__name__)


# ######
# TO DO: this is all blocking, oops
# ######

@app.route("/next_tile", methods=["GET"])
async def next_tile():
    """
    return the next tile
    """
    jd = float(request.args["jd"])

    sched = await wrapBlocking(Atomic)

    await wrapBlocking(sched.prepare_for_night, np.floor(jd))

    tile_id = await wrapBlocking(sched.next_tile, jd)

    next_tile = {"tile_id": int(tile_id)}

    return jsonify(next_tile)


@app.route("/cals", methods=["GET"])
async def cals():
    """
    return the next tile
    """
    tile_id = int(request.args["tile_id"])

    cals = await wrapBlocking(Cals, tile_id)
    skies = await wrapBlocking(cals.choose_skies)
    standards = await wrapBlocking(cals.choose_standards)

    cal_dict = {"tile_id": tile_id,
                "sky_pks": [int(s) for s in skies],
                "standard_pks": [int(s) for s in standards]}

    return jsonify(cal_dict)
