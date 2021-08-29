#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-17
# @Filename: region.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-13 11:32:39

import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.patches
import matplotlib.path
import matplotlib.transforms
import numpy
from copy import deepcopy

from spherical_geometry import polygon as sp
from lvmsurveysim.utils import plot as lvm_plot
from lvmsurveysim.exceptions import LVMSurveyOpsError, LVMSurveyOpsWarning

from . import _VALID_FRAMES


__all__ = ['SkyRegion']


# if we want to inherit: 
#super(SubClass, self).__init__('x')

class SkyRegion(object):
    def __init__(self, typ, coords, **kwargs):
        print(typ, coords, kwargs)
        self.region_type = typ
        self.frame = kwargs['frame']

        if typ == 'rectangle':

            self.center = coords
            width = kwargs['width'] / numpy.cos(numpy.deg2rad(coords[1]))
            height = kwargs['height']
            x0 = - width / 2.
            x1 = + width / 2.
            y0 = - height / 2.
            y1 = + height / 2.
            x, y = self._rotate_coords([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], kwargs['pa'])
            y += self.center[1]
            x += self.center[0]
            self.region =  sp.SphericalPolygon.from_radec(x, y, center=self.center, degrees=True)

        elif typ == 'circle':

            self.center = coords
            r = kwargs['r']
            k = int(numpy.max([numpy.floor(numpy.sqrt(r * 20)), 24]))
            x = numpy.array(list(reversed([r * numpy.cos(2.0*numpy.pi/k * i) for i in range(k+1)])))
            y = numpy.array(list(reversed([r * numpy.sin(2.0*numpy.pi/k * i) for i in range(k+1)])))
            y += self.center[1]
            x = x / numpy.cos(numpy.deg2rad(y)) + self.center[0]
            self.region = sp.SphericalPolygon.from_radec(x, y, center=self.center, degrees=True)
            # self.region = sp.SphericalPolygon.from_cone(coords[0], coords[1], kwargs['r'])
            # self.center = coords

        elif typ == 'ellipse':

            self.center = coords
            a, b = kwargs['a'], kwargs['b']
            k = int(numpy.max([numpy.floor(numpy.sqrt(((a + b) / 2) * 20)), 24]))
            x = list(reversed([a * numpy.cos(2.0*numpy.pi/k * i) for i in range(k+1)]))
            y = list(reversed([b * numpy.sin(2.0*numpy.pi/k * i) for i in range(k+1)]))
            x, y = self._rotate_coords(x, y, kwargs['pa'])
            y += self.center[1]
            x = x / numpy.cos(numpy.deg2rad(y)) + self.center[0]
            self.region = sp.SphericalPolygon.from_radec(x, y, center=self.center, degrees=True)

        elif typ == 'polygon':

            x, y = self._rotate_vertices(numpy.array(coords), 0.0)
            self.center = [numpy.average(x), numpy.average(y)]
            x -= self.center[0]
            x = x / numpy.cos(numpy.deg2rad(y)) + self.center[0]
            self.region = sp.SphericalPolygon.from_radec(x, y, center=self.center, degrees=True)

        else:
            raise LVMSurveyOpsError('Unknown region type '+typ)

    def vertices(self):
        i = self.region.to_lonlat()
        return numpy.array(next(i)).T        

    def bounds(self):
        x, y = next(self.region.to_lonlat())
        return numpy.min(x), numpy.min(y), numpy.max(x), numpy.max(y)

    def centroid(self):
        return self.center

    def intersects_poly(self, other):
        return self.region.intersects_poly(other.region)

    def contains_point(self, x, y):
        return self.region.contains_lonlat(x, y, degrees=True)

    def icrs_region(self):
        r2 = deepcopy(self)
        if self.frame == 'icrs':
            return r2
        else:
            r2.frame = 'icrs'
            x, y = next(self.region.to_lonlat())
            c = SkyCoord(self.center[0]*u.deg, self.center[1]*u.deg).transform_to('icrs')
            s = SkyCoord(x*u.deg, y*u.deg, frame=self.frame).transform_to('icrs')
            r2.center = [c.ra.deg, c.dec.deg]
            r2.region = sp.SphericalPolygon.from_radec(s.ra.deg, s.dec.deg, degrees=True)
            return r2

    def icrs_region_refine(self):
        r2 = deepcopy(self)
        if self.frame == 'icrs':
            return r2
        else:
            r2.frame = 'icrs'
            x, y = next(self.region.to_lonlat())
            x, y = self.polygon_perimeter(x, y)
            c = SkyCoord(self.center[0]*u.deg, self.center[1]*u.deg).transform_to('icrs')
            s = SkyCoord(x*u.deg, y*u.deg, frame=self.frame).transform_to('icrs')
            r2.center = [c.ra.deg, c.dec.deg]
            r2.region = sp.SphericalPolygon.from_radec(s.ra.deg, s.dec.deg, degrees=True)
            return r2

    @classmethod
    def polygon_perimeter(cls, x, y, n=1.0, min_points=5):
        """ x and y are numpy type arrays. Function returns perimiter values every n-degree in length"""
        x_perimeter = numpy.array([])
        y_perimeter = numpy.array([])
        for x1,x2,y1,y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
            # Calculate the length of a segment, hopefully in degrees
            dl = ((x2-x1)**2 + (y2-y1)**2)**0.5

            n_dl = numpy.max([int(dl/n), min_points])
            
            if x1 != x2:
                m = (y2-y1)/(x2-x1)
                b = y2 - m*x2

                interp_x = numpy.linspace(x1, x2, num=n_dl, endpoint=False)
                interp_y = interp_x * m + b
            
            else:
                interp_x = numpy.full(n_dl, x1)
                interp_y = numpy.linspace(y1,y2, n_dl, endpoint=False)

            x_perimeter = numpy.append(x_perimeter, interp_x)
            y_perimeter = numpy.append(y_perimeter, interp_y)
        return x_perimeter, y_perimeter


    def plot(self, ax=None, projection='rectangular', return_patch=False, **kwargs):
        """Plots the region.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to use. If `None`, new axes will be created.
        projection (str):
            The projection to use. At this time, only ``rectangular`` and
            ``mollweide`` are accepted.
        return_patch (bool):
            If True, returns the
            `matplotlib patch <https://matplotlib.org/api/patches_api.html>`_
            for the region.
        kwargs (dict):
            Options to be passed to matplotlib when creating the patch.

        Returns
        -------
        Returns the matplotlib ~matplotlib.axes.Axes` object for this plot.
        If not specified, the default plotting styles will be used.
        If ``return_patch=True``, returns the patch as well.

        """

        if ax is None:
            __, ax = lvm_plot.get_axes(projection=projection, frame=self.frame)

        coords = self.vertices()

        poly = matplotlib.path.Path(coords, closed=True)
        poly_patch = matplotlib.patches.PathPatch(poly, **kwargs)

        poly_patch = ax.add_patch(poly_patch)

        if projection == 'rectangular':
            ax.set_aspect('equal', adjustable='box')

            min_x, min_y = coords.min(0)
            max_x, max_y = coords.max(0)

            padding_x = 0.1 * (max_x - min_x)
            padding_y = 0.1 * (max_y - min_y)

            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)

        elif projection == 'mollweide':

            poly_patch = lvm_plot.transform_patch_mollweide(ax, poly_patch, patch_centre=self.center[0])

        if return_patch:
            return ax, poly_patch
        else:
            return ax

    @classmethod
    def _rotate_vertices(cls, vertices, pa):
        sa, ca = numpy.sin(numpy.deg2rad(pa)), numpy.cos(numpy.deg2rad(pa))
        R = numpy.array([[ca, -sa], [sa, ca]])
        return numpy.dot(R, vertices.T).T 

    @classmethod
    def _rotate_coords(cls, x, y, pa):
        rot = cls._rotate_vertices(numpy.array([x, y]).T, pa)
        xyprime = rot.T
        return xyprime[0,:], xyprime[1,:]
