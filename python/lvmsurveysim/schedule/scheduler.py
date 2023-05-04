#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation

import lvmsurveysim.target
from lvmsurveysim.schedule.plan import ObservingPlan, get_sun_moon_data
from lvmsurveysim import config
from lvmsurveysim.exceptions import LVMSurveyOpsError
from lvmsurveysim.schedule.altitude_calc import AltitudeCalculator

import skyfield.api
from lvmsurveysim.utils import shadow_height_lib

if os.getenv("OBSERVATORY") == "LCO":
    # mostly helps for sims
    from lvmsurveysim.schedule.opsdb import OpsDB

np.seterr(invalid='raise')


__all__ = ['Scheduler']


class Scheduler(object):
    """Selects optimal tile from a list of targets (tile database) at a given JD

    A typical usage scenario might look like:

    >>> plan = ObservingPlan(...)
    >>>    tiledb = TileDB.load('lco_tiledb')
    >>>    scheduler = Scheduler(plan)

    >>>    # observed exposure time for each pointing
    >>>    observed = np.zeros(len(tiledb), dtype=np.float)

    >>>    # range of dates for the survey
    >>>    dates = range(np.min(plan['JD']), np.max(plan['JD']) + 1)

    >>>    for jd in dates:
    >>>        scheduler.prepare_for_night(jd, plan, tiledb)
    >>>        current_jd = now()
    >>>        while current_jd < scheduler.morning_twi:
    >>>            observed_idx, current_lst, hz, alt, lunation = scheduler.get_optimal_tile(current_jd, observed)
    >>>            if observed_idx == -1:
    >>>                NOTHING TO DO
    >>>            else:
    >>>                RECORD OBSERVATION of the tile
    >>>            current_jd = now()

    Parameters
    ----------
    observing_plan : `.ObservingPlan`
        The `.ObservingPlan` to use. Contains dates and sun/moon data for the 
        duration of the survey as well as Observatory data.
        The plan is only used to initialize the shadow height calculator with location
        information of the observatory. The plan is not stored. The plan used for
        scheduling is passed in `prepare_for_night()`.

    """

    def __init__(self, observing_plan):

        assert isinstance(observing_plan, ObservingPlan), 'observing_plan is not an instance of ObservingPlan.'
        self.observatory = observing_plan.observatory
        self.lon = observing_plan.location.lon.deg
        self.lat = observing_plan.location.lat.deg

        self.zenith_avoidance = config['scheduler']['zenith_avoidance']

        if os.getenv("OBSERVATORY") == "LCO":
            load = skyfield.api.Loader("/home/sdss5/config/skyfield-data")
        else:
            load = skyfield.api.Loader(os.path.expanduser("~/.config"))
        eph = load('de421.bsp')
        self.shadow_calc = shadow_height_lib.shadow_calc(observatory_name=self.observatory,
                                observatory_elevation=observing_plan.location.height,
                                observatory_lat=self.lat, observatory_lon=self.lon,
                                eph=eph, earth=eph['earth'], sun=eph['sun'])

    def __repr__(self):
        return (f'<Scheduler (observing_plans={self.observatory})> ')

    def prepare_for_night(self, jd, plan, tiledb):
        """Initializes and caches various quantities to use for scheduling observations
        for a given single JD.

        This method MUST be called once for each JD prior to (repeatedly)
        calling `get_optimal_tile()` for a sequence of JDs between dusk and dawn the same day.

        Parameters
        ----------
        jd : int
            The Julian Date of the night to schedule. Must be included in ``plan``.
        plan : .ObservingPlan
            The observing plan containing at least the night corresponding to jd.
            Must be for the same observatory as the plan passed in ctor.
        tiledb : .TileDB
            The tile database for the night (or survey)

        """

        # No. Just make tiledb an astropy table
        # assert isinstance(tiledb, TileDB), 'tiledb must be a lvmsurveysim.schedule.tiledb.TileDB instances.'
        self.tiledb = tiledb

        assert isinstance(plan, ObservingPlan), \
            'one of the items in observing_plans is not an instance of ObservingPlan.'

        self.maxpriority = max([t for t in tiledb["target_priority"].data])

        night_plan = plan[plan['JD'] == jd]
        self.evening_twi = night_plan['evening_twilight'][0]
        self.morning_twi = night_plan['morning_twilight'][0]

        # Get the Moon lunation and distance to targets, assume it is constant
        # for the night for speed.
        self.lunation = night_plan['moon_phase'][0]

        ra = self.tiledb['ra'].data
        dec = self.tiledb['dec'].data

        self.moon_to_pointings = lvmsurveysim.utils.spherical.great_circle_distance(
                                 night_plan['moon_ra'], night_plan['moon_dec'], ra, dec)

        # set the coordinates to all targets in shadow height calculator
        self.shadow_calc.set_coordinates(ra, dec)

        # Fast altitude calculator
        self.ac = AltitudeCalculator(ra, dec, self.lon, self.lat)

        # convert airmass to altitude, we'll work in altitude space for efficiency
        self.min_alt_for_target = 90.0 - np.rad2deg(np.arccos(1.0 / self.tiledb['airmass_limit'].data))

        # Select targets that are above the max airmass and with good
        # moon avoidance.
        self.moon_ok = (self.moon_to_pointings > self.tiledb['moon_distance_limit'].data)\
                     & (self.lunation <= self.tiledb['lunation_limit'].data)

    def get_optimal_tile(self, jd, observed, done=None):
        """Returns the next tile to observe at a given (float) jd.

        jd must be between the times of evening and morning twilight on the id
        previously passed to `prepare_for_night()`.

        `get_optimal_tile()` can be called repeatedly for times during that night.

        The scheduling algorithm first selects all tiles that fulfil the constraints on
        lunation, zenith distance, airmass, moon distance and shadow height.
        If there are tiles that have > 0 but less than the required visits, choose those
        otherwise choose from the target with the highest priority.
        If the target's tiling strategy assigns tile priorities, choose the tile with the
        hightest priority otherwise choose the one with the highest airmass.

        Parameters
        ----------
        jd : float
            The Julian Date to schedule. Must be between evening and morning twilight according
            to the observing plan.

        observed : ~np.array
            Same length as len(tiledb).
            Array containing the exposure time already executed for each tile in the tiledb.
            This is used to keep track of which tiles need additional time and which are completed.

            This record is kept by the caller and passed in, since an observation might fail 
            in real life. So accounting of actual observing time spent on target must be the
            responsibility of the caller.

        done : np.array
            Same length as len(tiledb).
            Array containing boolean done or not per tile

        Returns
        -------
        observed_idx : int
            Index into the tiledb of the tile to be observed next.
        current_lst : float
            The LST of the observation
        hz : float
            The shadow height for the observation
        alt : float
            The altitude of the observation
        lunation : float
            The lunation at time of the observation

        """

        if jd >= self.morning_twi or jd < self.evening_twi:
            raise LVMSurveyOpsError(f'the time {jd} is not between {self.evening_twi} and {self.morning_twi}.')

        tdb = self.tiledb
        if len(tdb) != len(observed):
            raise LVMSurveyOpsError(f'length of tiledb {len(tdb)} != length of observed array {len(observed)}.')

        # Get current LST
        lst = lvmsurveysim.utils.spherical.get_lst(jd, self.lon)

        # advance shadow height calculator to current time
        self.shadow_calc.update_time(jd=jd)

        # Get the altitude at the start and end of the proposed exposure.
        alt_start = self.ac(lst=lst)
        alt_end = self.ac(lst=(lst + (tdb['visit_exptime'].data / 3600.)))

        # avoid the zenith!
        alt_ok = (alt_start < (90 - self.zenith_avoidance)) & (alt_end < (90 - self.zenith_avoidance))

        # Gets valid airmasses (but we're working in altitude space)
        airmass_ok = ((alt_start > self.min_alt_for_target) & (alt_end > self.min_alt_for_target))

        # Gets pointings that haven't been completely observed
        if done is None:
            done = observed >= tdb['total_exptime'].data

        # Creates a mask of viable pointings with correct Moon avoidance,
        # airmass, zenith avoidance and that have not been completed.
        valid_mask = alt_ok & self.moon_ok & airmass_ok & ~done

        # calculate shadow heights, but only for the viable pointings since it is a costly computation
        hz = np.full(len(alt_ok), 0.0)
        hz_valid = self.shadow_calc.get_heights(return_heights=True, mask=valid_mask, unit="km")
        hz[valid_mask] = hz_valid
        hz_ok = (hz > tdb['hz_limit'].data)

        # add shadow height to the viability criteria of the pointings to create the final 
        # subset that are candidates for observation
        valid_idx = np.where(valid_mask & hz_ok)[0]

        # If there's nothing to observe, return -1
        if len(valid_idx) == 0:
            return -1, lst, 0, 0, self.lunation

        # Find observations that have nonzero exposure but are incomplete
        incomplete = (observed > 0) & (~done)

        target_priorities = tdb['target_priority'].data
        tile_priorities = tdb['tile_priority'].data

        # Gets the coordinates, altitudes, and priorities of possible pointings.
        valid_alt = alt_start[valid_idx]
        valid_priorities = target_priorities[valid_idx]
        valid_incomplete = incomplete[valid_idx]
        valid_tile_priorities = tile_priorities[valid_idx]

        # Give incomplete observations the highest priority, imitating a high-priority target,
        # that makes sure these are completed first in all visible targets
        valid_priorities[valid_incomplete] = self.maxpriority + 1

        # Loops starting with targets with the highest priority (lowest numerical value).
        for priority in np.flip(np.unique(valid_priorities), axis=0):

            # Gets the indices that correspond to this priority (note that
            # these indices correspond to positions in valid_idx, not in the
            # master list).
            valid_priority_idx = np.where(valid_priorities == priority)[0]

            # If there's nothing to do at the current priority, try the next lower
            if len(valid_priority_idx) == 0:
                continue

            # select all pointings with the current target priority
            valid_alt_target_priority = valid_alt[valid_priority_idx]
            valid_alt_tile_priority = valid_tile_priorities[valid_priority_idx]

            # Find the tiles with the highest tile priority
            max_tile_priority = np.max(valid_alt_tile_priority)
            high_priority_tiles = np.where(valid_alt_tile_priority == max_tile_priority)[0]

            # Gets the pointing with the highest altitude * shadow height
            obs_alt_idx = (valid_alt_target_priority[high_priority_tiles]).argmax()
            #obs_alt_idx = (hz[valid_priority_idx[high_priority_tiles]] * valid_alt_target_priority[high_priority_tiles]).argmax()
            #obs_alt_idx = hz[valid_priority_idx[high_priority_tiles]].argmax()
            obs_tile_idx = high_priority_tiles[obs_alt_idx]
            obs_alt = valid_alt_target_priority[obs_tile_idx]

            # Gets the index of the pointing in the master list.
            observed_idx = valid_idx[valid_priority_idx[obs_tile_idx]]

            return observed_idx, lst, hz[observed_idx], obs_alt, self.lunation

        assert False, "Unreachable code!"
        return -1, 0   # should never be reached


class Atomic(object):
    """A basic, constrained wrapper around Scheduler to fill "live" or
       "on the fly" requests.
    """

    def __init__(self):
        survey_start = 2459458
        survey_end = 2460856

        self.observing_plan = ObservingPlan(survey_start, survey_end, observatory='LCO')
        self._tiledb = None
        self._scheduler = None
        self._history = None
        self._done = None

    @property
    def tiledb(self):
        if self._tiledb is None:
            self._tiledb = OpsDB.load_tiledb()
        return self._tiledb

    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = Scheduler(self.observing_plan)
        return self._scheduler

    @property
    def history(self):
        if self._history is None:
            # full_hist = OpsDB.load_history()
            # hist_in_order = [full_hist[tile_id] for tile_id in self.tiledb["tile_id"]]
            self._history = np.zeros(len(self.tiledb), dtype=np.float64)
        return self._history

    @property
    def done(self):
        if self._done is None:
            # full_done = OpsDB.load_history()
            # done_in_order = [full_done[tile_id] for tile_id in self.tiledb["tile_id"]]
            self._done = np.zeros(len(self.tiledb), dtype=bool)
        return self._done

    def prepare_for_night(self, jd):
        self._tiledb = None
        self._history = None

        self.scheduler.prepare_for_night(jd, self.observing_plan, self.tiledb)

    def next_tile(self, jd):
        idx, current_lst, hz, alt, lunation = \
            self.scheduler.get_optimal_tile(jd, self.history, self.done)

        tileid = self.tiledb['tile_id'].data[idx]
        exptime = self.tiledb['visit_exptime'].data[idx]
        self.observed[idx] += exptime

        return tileid


class Cals(object):
    """
    A convenience class to choose skies and standards
    """

    def __init__(self, tile_id=None, ra=None, dec=None, jd=None):
        if tile_id:
            self.ra, self.dec = OpsDB.retrieve_tile_ra_dec(tile_id)
        else:
            self.ra = ra
            self.dec = dec
        self.jd = jd
        self._skies = None
        self._standards = None
        self.observatory = "LCO"
        self.location = EarthLocation.of_site(self.observatory)
        self.lon = self.location.lon.deg
        self.lat = self.location.lat.deg

        load = skyfield.api.Loader("/home/sdss5/config/skyfield-data")
        eph = load('de421.bsp')

        self.shadow_calc = shadow_height_lib.shadow_calc(observatory_name=self.observatory,
                              observatory_elevation=self.location.height,
                              observatory_lat=self.lat, observatory_lon=self.lon,
                              eph=eph, earth=eph['earth'], sun=eph['sun'])

    @property
    def skies(self):
        if self._skies is None:
            self._skies = OpsDB.load_sky(ra=self.ra, dec=self.dec,
                                         radius=10)
        return self._skies

    @property
    def standards(self):
        if self._standards is None:
            self._standards = OpsDB.load_standard(ra=self.ra, dec=self.dec,
                                                  radius=10)
        return self._standards

    def skyCostFunc(self):
        if self.jd is None:
            now = Time.now()
            now.format = "jd"
            self.jd = now.value
        lst = lvmsurveysim.utils.spherical.get_lst(self.jd, self.lon)

        targ_dist = self.center_distance(self.skies["ra"].data,
                                         self.skies["dec"].data)
        self.shadow_calc.set_coordinates(self.skies["ra"].data,
                                         self.skies["dec"].data)
        hz = self.shadow_calc.get_heights(return_heights=True, jd=self.jd)
        ac = AltitudeCalculator(self.skies["ra"].data,
                                self.skies["dec"].data,
                                self.lon, self.lat)
        am = 1. / np.sin(np.pi / 180. * ac(lst=lst))
        spos, mpos, k = get_sun_moon_data(self.jd, location=self.location)
        moon_sky_dist = lvmsurveysim.utils.spherical.great_circle_distance(
                           mpos.ra.deg, mpos.dec.deg,
                           self.skies["ra"].data, self.skies["dec"].data)
        moon_targ_dist = lvmsurveysim.utils.spherical.great_circle_distance(
                           mpos.ra.deg, mpos.dec.deg, self.ra, self.dec)

        # print("hz", len(hz), hz)
        # print("am", len(am), am)
        # print("moon_sky_dist", len(moon_sky_dist), moon_sky_dist)
        # print("moon_targ_dist", len(moon_targ_dist), moon_targ_dist)
        # print("targ_dist", len(targ_dist), targ_dist)

        min_diff = np.min(np.abs(moon_sky_dist - moon_targ_dist))

        # for z, a, ms, t in zip(hz, am, moon_sky_dist, targ_dist):
        #     print(f"{z:.1f} {a:.2f} {ms:.1f} {t:.1f}")
        #     print(f"{(5000/(500-z)):.1f} {a*10:.2f} {10*float(np.abs(ms - moon_targ_dist)/min_diff):.1f}")
        #     print((5000/(500-z)) + a*10 + 10*np.abs(ms - moon_targ_dist)/min_diff + t)

        return (5000/(500-hz)) + am*10 +\
               2*(np.abs(moon_sky_dist - moon_targ_dist) - min_diff) +\
               targ_dist

    def center_distance(self, ra, dec):
        assert len(ra) == len(dec), "ra and dec must be same length"
        dist = lvmsurveysim.utils.spherical.great_circle_distance(
                  self.ra, self.dec, ra, dec)
        return dist

    def choose_skies(self, N=2):
        # dist = self.center_distance(self.skies["ra"].data,
        #                             self.skies["dec"].data)
        # first_N = np.argsort(dist)[:N]

        cost = self.skyCostFunc()
        first_N = np.argsort(cost)[:N]
        return self.skies["pk"][first_N]

    def choose_standards(self, N=12):
        dist = self.center_distance(self.standards["ra"].data,
                                    self.standards["dec"].data)
        first_N = np.argsort(dist)[:N]
        return self.standards["pk"][first_N]
