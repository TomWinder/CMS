############################################################################
############## Scripts for Generation of Travel-Time LUT ###################
############################################################################
#   Adaptations from IntraSeis LUT generation and saving.
#
# ##########################################################################
# ---- Import Packages -----
import math
import warnings
from copy import copy

import numpy as np
import pyproj
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

import subprocess
import os
import pandas as pd
import pickle

# ---- Coordinate transformations ----

def _cart2sph_np_array(xyz):
    # theta_phi_r = _cart2sph_np_array(xyz)
    tpr = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    tpr[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
    tpr[:, 1] = np.arctan2(xyz[:, 2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
    tpr[:, 2] = np.sqrt(xy + xyz[:, 2] ** 2)
    return tpr


def _cart2sph_np(xyz):
    # theta_phi_r = _cart2sph_np(xyz)
    if xyz.ndim == 1:
        tpr = np.zeros(3)
        xy = xyz[0] ** 2 + xyz[1] ** 2
        tpr[0] = np.arctan2(xyz[1], xyz[0])
        tpr[1] = np.arctan2(xyz[2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
        tpr[2] = np.sqrt(xy + xyz[2] ** 2)
    else:
        tpr = np.zeros(xyz.shape)
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        tpr[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
        tpr[:, 1] = np.arctan2(xyz[:, 2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
        tpr[:, 2] = np.sqrt(xy + xyz[:, 2] ** 2)
    return tpr


def _sph2cart_np(tpr):
    # xyz = _sph2cart_np(theta_phi_r)
    if tpr.ndim == 1:
        xyz = np.zeros(3)
        xyz[0] = tpr[2] * np.cos(tpr[1]) * np.cos(tpr[0])
        xyz[1] = tpr[2] * np.cos(tpr[1]) * np.sin(tpr[0])
        xyz[2] = tpr[2] * np.sin(tpr[1])
    else:
        xyz = np.zeros(tpr.shape)
        xyz[:, 0] = tpr[:, 2] * np.cos(tpr[:, 1]) * np.cos(tpr[:, 0])
        xyz[:, 1] = tpr[:, 2] * np.cos(tpr[:, 1]) * np.sin(tpr[:, 0])
        xyz[:, 2] = tpr[:, 2] * np.sin(tpr[:, 1])
    return xyz


def _coord_transform_np(p1, p2, loc):
    xyz = np.zeros(loc.shape)
    if loc.ndim == 1:
        xyz[0], xyz[1], xyz[2] = pyproj.transform(p1, p2, loc[0], loc[1], loc[2])
    else:
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = pyproj.transform(p1, p2, loc[:, 0], loc[:, 1], loc[:, 2])
    return xyz

def _proj_wgs84():
    return pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")  # "+init=EPSG:4326"


def _proj_nad27():
    return pyproj.Proj("+proj=longlat +ellps=clrk66 +datum=NAD27 +no_defs")  # "+init=EPSG:4267"


def _proj_wgs84_utm(longitude):
    zone = (int(1 + math.fmod((longitude + 180.0) / 6.0, 60)))
    return pyproj.Proj("+proj=utm +zone={0:d} +datum=WGS84 +units=m +no_defs".format(zone))


# ------- Class definition of the structure and manipulation of grid -------------
class Grid3D:
    def __init__(self, center=np.array([10000.0, 10000.0, -5000.0]), cell_count=np.array([51, 51, 31]),
                 cell_size=np.array([30.0, 30.0, 30.0]),
                 azimuth=0.0, dip=0.0, sort_order='C'):
        self._latitude = 51.4826
        self._longitude = 0.0077
        self._coord_proj = None
        self._grid_proj = None
        self._grid_center = None
        self._cell_count = None
        self._cell_size = None
        self.grid_center = center
        self.cell_count = cell_count
        self.cell_size = cell_size
        self.grid_azimuth = azimuth
        self.grid_dip = dip
        self.sort_order = sort_order

    @property
    def grid_center(self):
        return self._grid_center

    @grid_center.setter
    def grid_center(self, value):
        value = np.array(value, dtype='float64')
        assert (value.shape == (3,)), 'Grid center must be [x, y, z] array.'
        self._grid_center = value
        self._update_coord()

    @property
    def grid_proj(self):
        return self._grid_proj

    @grid_proj.setter
    def grid_proj(self, value):
        self._grid_proj = value
        self._update_grid_center()

    @property
    def coord_proj(self):
        return self._coord_proj

    @coord_proj.setter
    def coord_proj(self, value):
        self._coord_proj = value
        self._update_coord()

    @property
    def cell_count(self):
        return self._cell_count

    @cell_count.setter
    def cell_count(self, value):
        value = np.array(value, dtype='int32')
        if value.size == 1:
            value = np.repeat(value, 3)
        else:
            assert (value.shape == (3,)), 'Cell count must be [nx, ny, nz] array.'
        assert (np.all(value > 0)), 'Cell count must be greater than [0]'
        self._cell_count = value

    @property
    def cell_size(self):
        return self._cell_size

    @cell_size.setter
    def cell_size(self, value):
        value = np.array(value, dtype='float64')
        if value.size == 1:
            value = np.repeat(value, 3)
        else:
            assert (value.shape == (3,)), 'Cell size must be [dx, dy, dz] array.'
        assert (np.all(value > 0)), 'Cell size must be greater than [0]'
        self._cell_size = value

    @property
    def elevation(self):
        return self._grid_center[2]

    @elevation.setter
    def elevation(self, value):
        self._grid_center[2] = value

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    def set_proj(self, coord_proj=None, grid_proj=None):
        if coord_proj:
            self._coord_proj = coord_proj
        if grid_proj:
            self._grid_proj = grid_proj
        self._update_coord()

    def get_grid_proj(self):
        if self._grid_proj is None:
            warnings.warn("Grid Projection has not been set: Assuming WGS84")
            return _proj_wgs84_utm(self.longitude)
        else:
            return self._grid_proj

    def get_coord_proj(self):
        if self._coord_proj is None:
            warnings.warn("Coordinte Projection has not been set: Assuming WGS84")
            return _proj_wgs84()
        else:
            return self._coord_proj

    def _update_grid_center(self):
        if self._coord_proj and self._grid_proj and self._latitude and self._longitude:
            x, y = pyproj.transform(self._coord_proj, self._grid_proj, self._longitude, self._latitude)
            self._grid_center[0] = x
            self._grid_center[1] = y
            return True
        else:
            return False

    def _update_coord(self):
        if self._coord_proj and self._grid_proj:
            center = self._grid_center
            lat, lon = pyproj.transform(self._grid_proj, self._coord_proj, center[0], center[1])
            self._latitude = lat
            self._longitude = lon
            return True
        else:
            return False

    def set_lonlat(self, longitude=None, latitude=None, coord_proj=None, grid_proj=None):
        if coord_proj:
            self._coord_proj = coord_proj
        if grid_proj:
            self._grid_proj = grid_proj
        if latitude:
            self._latitude = latitude
        if longitude:
            self._longitude = longitude
        self._update_grid_center()

    def setproj_wgs84(self):
        self._coord_proj = _proj_wgs84()
        self._grid_proj = _proj_wgs84_utm(self.longitude)
        if not self._update_grid_center():
            self._update_coord()

    def xy2lonlat(self, x, y):
        return pyproj.transform(self.get_grid_proj(), self.get_coord_proj(), np.array(x), np.array(y))

    def lonlat2xy(self, lon, lat):
        return pyproj.transform(self.get_coord_proj(), self.get_grid_proj(), np.array(lon), np.array(lat))

    def local2global(self, loc):
        tpr = _cart2sph_np(loc - self._grid_center)
        tpr += [self.grid_azimuth, self.grid_dip, 0.0]
        return (_sph2cart_np(tpr) + self._grid_center)

    def global2local(self, loc):
        tpr = _cart2sph_np(loc - self._grid_center)
        tpr -= [self.grid_azimuth, self.grid_dip, 0.0]
        return (_sph2cart_np(tpr) + self._grid_center)

    def loc2xyz(self, loc):
        return self.local2global(self._grid_center + self._cell_size * (loc - (self._cell_count - 1) / 2))

    def xyz2loc(self, cord):
        return ((self.global2local(cord) - self._grid_center) / self._cell_size) + (self._cell_count - 1) / 2

    def loc2index(self, loc):
        return np.ravel_multi_index(loc, self._cell_count, mode='clip', order=self.sort_order)

    def index2loc(self, index):
        loc = np.vstack(np.unravel_index(index, self._cell_count, order=self.sort_order)).transpose()
        return loc

    def index2xyz(self, index):
        return self.loc2xyz(self.index2loc(index))

    def xyz2index(self, cord):
        return self.loc2index(self.xyz2loc(cord))

    def loc2coord(self, loc):
        lon, lat = self.xy2lonlat(loc[0], loc[1])
        return lon, lat, loc[2]

    def grid_origin(self):
        grid_size = (self._cell_count - 1) * self._cell_size
        return self.local2global(self._grid_center - grid_size / 2)

    def get_grid_xyz(self, cells='corner'):
        if cells == 'corner':
            lc = self._cell_count - 1
            ly, lx, lz = np.meshgrid([0, lc[1]], [0, lc[0]], [0, lc[2]])
            loc = np.c_[lx.flatten(), ly.flatten(), lz.flatten()]
            return self.loc2xyz(loc)
        else:
            lc = self._cell_count
            ly, lx, lz = np.meshgrid(np.arange(lc[1]), np.arange(lc[0]), np.arange(lc[2]))
            loc = np.c_[lx.flatten(), ly.flatten(), lz.flatten()]
            coord = self.loc2xyz(loc)
            lx = coord[:, 0].reshape(lc)
            ly = coord[:, 1].reshape(lc)
            lz = coord[:, 2].reshape(lc)
            return lx, ly, lz



# ------------ LUT Generation for the 3D LUT -------------

class LUT(Grid3D):
    '''
        Generating and Altering the Travel-Time LUT for


        maps            - Used later to apply Coalescence 4D data.
        _select_station - Selecting the stations to be used in the LUT
        decimate        - Downsample the intitial velocity model tables that are loaded before processing.
        get_station_xyz - Getting the stations relative x,y,z positions to the origin
        set_station     - Defining the station locations to be used

        ADDITON - Currently 'maps' stored in RAM. Need to use JSON or HDF5

    '''

    def __init__(self, center=np.array([10000.0, 10000.0, -5000.0]), cell_count=np.array([51, 51, 31]),
                 cell_size=np.array([30.0, 30.0, 30.0]), azimuth=0.0, dip=0.0):
        Grid3D.__init__(self, center, cell_count, cell_size, azimuth, dip)
        self.velocity_model = None
        self.station_data = None
        self.NLLoc_Path = None
        self._maps = dict()

    @property
    def maps(self):
        return self._maps

    @maps.setter
    def maps(self, maps):
        self._maps = maps


    def set_NLLocPATH(self,PATH=None):

        ''' Defining the PATH to the NLLoc Source files
                e.g. PATH = '~/Software/NonLinLoc/src' 
        '''
        self.NLLoc_Path = PATH
        # try:
        FNULL = open(os.devnull, 'w')
        OutputCode = subprocess.call(['{}/NLLoc'.format(PATH)], stdout=FNULL, stderr=FNULL)

        if OutputCode == 254:
            self.NLLoc_Path = PATH
            print('NLLoc src Path - {}'.format(PATH))
        else:
            print('Error Executing NLLoc Source Code - Please Check')



    def _select_station(self, station_data):
        if self.station_data is None:
            return station_data
        nstn = len(self.station_data)
        flag = np.array(np.zeros(nstn, dtype=np.bool))
        for i, stn in enumerate(self.station_data['Name']):
            if stn in station_data['Name']:
                flag[i] = True

    def decimate(self, ds, inplace=False):
        if not inplace:
            lut = copy(self)
            lut.maps = copy(lut.maps)
        else:
            lut = self
        ds = np.array(ds, dtype=np.int)
        cell_count = 1 + (lut.cell_count - 1) // ds
        c1 = (lut.cell_count - ds * (cell_count - 1) - 1) // 2
        cn = c1 + ds * (cell_count - 1) + 1
        center_cell = (c1 + cn - 1) / 2
        center = lut.loc2xyz(center_cell)
        lut.cell_count = cell_count
        lut.cell_size = lut.cell_size * ds
        lut.center = center
        maps = lut.maps
        if maps is not None:
            for id, map in maps.items():
                maps[id] = np.ascontiguousarray(map[c1[0]::ds[0], c1[1]::ds[1], c1[2]::ds[2], :])
        if not inplace:
            return lut

    def get_station_xyz(self, station=None):
        if station is not None:
            station = self._select_station(station)
            stn = self.station_data[station]
        else:
            stn = self.station_data
        x, y = self.lonlat2xy(stn['Longitude'], stn['Latitude'])
        coord = np.c_[x, y, stn['Elevation']]
        return coord

    def get_station_offset(self, station=None):
        coord = self.get_station_xyz(station)
        return coord - self.grid_center

    def get_values_at(self, loc, station=None):
        val = dict()
        for map in self.maps.keys():
            val[map] = self.get_value_at(map, loc, station)
        return val

    def get_value_at(self, map, loc, station=None):
        return self.interpolate(map, loc, station)

    def value_at(self, map, coord, station=None):
        loc = self.xyz2loc(coord)
        return self.interpolate(map, loc, station)

    def values_at(self, coord, station=None):
        loc = self.xyz2loc(coord)
        return self.get_values_at(loc, station)

    def get_interpolator(self, map, station=None):
        maps = self.fetch_map(map, station)
        nc = self._cell_count
        cc = (np.arange(nc[0]), np.arange(nc[1]), np.arange(nc[2]))
        return RegularGridInterpolator(cc, maps, bounds_error=False)

    def interpolate(self, map, loc, station=None):
        interp_fcn = self.get_interpolator(map, station)
        return interp_fcn(loc)

    def fetch_map(self, map, station=None):
        if station is None:
            return self.maps[map]
        else:
            station = self._select_station(station)
            return self.maps[map][..., station]

    def fetch_index(self, map, srate, station=None):
        maps = self.fetch_map(map, station)
        return np.rint(srate * maps).astype(np.int32)

    def set_station(self, loc, names=None, units='lon_lat_elev'):
        nstn = loc.shape[0]
        stn_data = pd.DataFrame(columns=['Longitude','Latitude','Elevation','Name'])
        if units == 'offset':
            stn_lon, stn_lat = self.xy2lonlat(loc[:, 0].astype('float') + self.grid_center[0], loc[:, 1].astype('float') + self.grid_center[1])
            stn_data['Longitude'] = stn_lon
            stn_data['Latitude'] = stn_lat
            stn_data['Elevation'] = loc[:, 2]
            stn_data['Name'] = loc[:,3]
        elif units == 'xyz':
            stn_lon, stn_lat = self.xy2lonlat(loc[:, 0], loc[:, 1])
            stn_data['Longitude'] = stn_lon
            stn_data['Latitude'] = stn_lat
            stn_data['Elevation'] = loc[:, 2]
            stn_data['Name'] = loc[:,3]
        elif units == 'lon_lat_elev':
            stn_data['Longitude'] = loc[:, 0]
            stn_data['Latitude'] = loc[:, 1]
            stn_data['Elevation'] = loc[:, 2]
            stn_data['Name'] = loc[:,3]
        elif units == 'lat_lon_elev':
            stn_data['Longitude'] = loc[:, 1]
            stn_data['Latitude'] = loc[:, 0]
            stn_data['Elevation'] = loc[:, 2]
            stn_data['Name'] = loc[:,3]
        self.station_data = stn_data


    def load_NLLoc(self,PATH=None):
        '''
            Structuring the LUT to read directily from a NonLinLoc Travel-Time table file. 
            This will read the file from the NonLinLoc format FILE.P.time or FILE.S.time 

            Additional checks needed:
                - Check if NonLinLoc correctly installed and added to PATH
                - Check that file exists
        '''

        # --------------- Defining the location to the Travel-Time Tables -----------------

        if self.NLLoc_Path == None:
            print('Please specify a NLLoc Source path using - set_NLLocPATH')

        elif PATH == None:
            print('Please supply a host path for the NonLinLoc Travel-Time Tables')

        else:
            # Constructing the correct file format for the NonLinLoc files
            self.station_data['NNLoc_PTime_PATH'] = '{}.P.'.format(PATH) + self.station_data['Name'] + '.time'
            self.station_data['NNLoc_STime_PATH'] = '{}.S.'.format(PATH) + self.station_data['Name'] + '.time'

            # Add a checking stage to see if these load.

        # --------------- Redefining the LUT information based on .hdr files  -----------------
        # ~~~ Add this later, currently will exspect the user to be consistent. 


    def create_NLLoc(lut, PATH):
        '''
            Creating the required Travel-Time table using the methods in NonLinLoc.

            This would just run the standard NonLinLoc exacutable from Python so not imperative.
        
        '''

def compute_homogeneous_lut(lut, vp, vs):
    '''
        Generates a Homogeneous Velocity model.

        lut   - LUT to determine travel-time on
        vp    - np.array of P-wave velocity at corresponding zp value
        vs    - np.array of S-wave velocity at corresponding zp value

    '''
    rloc = lut.get_station_xyz()
    gx, gy, gz = lut.get_grid_xyz(cells='all')
    nstn = rloc.shape[0]
    ncell = lut.cell_count
    map_p1 = np.zeros(np.r_[ncell, nstn])
    map_s1 = np.zeros(np.r_[ncell, nstn])
    for stn in range(nstn):
        dx = gx - float(rloc[stn, 0])
        dy = gy - float(rloc[stn, 1])
        dz = gz - float(rloc[stn, 2])
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        map_p1[..., stn] = (dist / vp)
        map_s1[..., stn] = (dist / vs)
    return {'TIME_P': map_p1, 'TIME_S': map_s1}

    
    

def compute_fd1d_lut(lut, zp, vp, vs, delta=None, epsilon=None, gamma=None, dx=10.0):
    '''

        IMPORT : Currently this will not work in the current situation since the C program exacutable is missing.

        Generates a 1D finite-difference travel tables.

        lut   - The CMS LUT generted through cmslut.LUT
        zp    - np.array of depth
        vp    - np.array of P-wave velocity at corresponding zp value
        vs    - np.array of S-wave velocitu at corresponding zp value
        delta - Optional: Thomsen's EPSILON paramete
        gama  - Optional: Thomsen's GAMMA parameter
        dx    - Optional: Thomsen's EPSILON paramete


        ADDITION NEEDED -  Sequential loading using pd.HDFStore


    '''
    stn = lut.get_station_xyz()
    coord = lut.get_grid_xyz()
    minc = np.min([np.min(coord, 0), np.min(stn, 0)], 0)
    maxc = np.max([np.max(coord, 0), np.max(stn, 0)], 0)
    rngc = maxc - minc
    xoff = np.sqrt(rngc[0] * rngc[0] + rngc[1] * rngc[1])

    grdz = np.arange(minc[2] - 10 * dx, maxc[2] + 10 * dx, dx)
    grof = np.arange(0, xoff + 10 * dx, dx)
    gvp = np.interp(-grdz, -zp, vp)
    gvs = np.interp(-grdz, -zp, vs)

    nx = len(grof)
    nz = len(grdz)
    nstn = stn.shape[0]

    vmdl = ilib.vti_model(gvp, gvs, delta=None, epsilon=None, gamma=None)

    ix, iy, iz = lut.get_grid_xyz(cells='all')
    ttp1 = np.zeros(ix.shape + (nstn,))
    ttsv = np.zeros(ix.shape + (nstn,))
    ttsh = np.zeros(ix.shape + (nstn,))

    for i in range(nstn):
        print("Generating 1D Travel-Time Table - {}".format(i))
        src_z = np.argmin(np.abs(stn[i, 2] - grdz))
        ofx = ix - stn[i, 0]
        ofy = iy - stn[i, 1]
        ioff = np.sqrt(ofx * ofx + ofy * ofy)

        tt, _, _ = ilib.fdtt_compute(0, vmdl, nx, nz, dx, src_z)
        ilab = RectBivariateSpline(grdz, grof, tt)
        ttp1[..., i] = ilab.ev(iz, ioff)

        tt, _, _ = ilib.fdtt_compute(1, vmdl, nx, nz, dx, src_z)
        ilab = RectBivariateSpline(grdz, grof, tt)
        ttsv[..., i] = ilab.ev(iz, ioff)

        tt, _, _ = ilib.fdtt_compute(2, vmdl, nx, nz, dx, src_z)
        ilab = RectBivariateSpline(grdz, grof, tt)
        ttsh[..., i] = ilab.ev(iz, ioff)

        # SHOULD PROBABLY SAVE TO FILE ON LARGE LUT's

    lut.maps = {'TIME_P1': ttp1, 'TIME_SV': ttsv, 'TIME_SH': ttsh}

    return grdz, grof, ttp1, ttsv, ttsh
