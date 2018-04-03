############################################################################
############## Scripts for Scanning and Coalescence of Data ################
############################################################################
# ---- Import Packages -----

#import hjson as json
import numpy as np
#from obspy.signal.trigger import classic_sta_lta
from CMS.core.time import UTCDateTime
from datetime import datetime
from obspy.signal.trigger import classic_sta_lta
from scipy.signal import butter, lfilter
import CMS.core.cmslib as ilib


import os.path as path

# ----- Useful Functions -----
def _utc_datetime(dtime):
    return UTCDateTime(dtime)

def onset(sig, stw, ltw):
    # assert isinstance(snr, object)
    nchan, nsamp = sig.shape
    snr = np.copy(sig)
    for ch in range(0, nchan):
        snr[ch, :] = classic_sta_lta(sig[ch, :], stw, ltw)
        np.clip(1+snr[ch,:],0.8,np.inf,snr[ch, :])
        np.log(snr[ch, :], snr[ch, :])
    return snr


def filter(sig,srate,lc,hc,order=3):
    b1, a1 = butter(order, [2.0*lc/srate, 2.0*hc/srate], btype='band')
    nchan, nsamp = sig.shape
    fsig = np.copy(sig)
    #sig = detrend(sig)
    for ch in range(0, nchan):
        fsig[ch,:] = fsig[ch,:] - fsig[ch,0]
        fsig[ch,:] = lfilter(b1, a1, fsig[ch,:])
    return fsig


def _time_str(daten=None):
    if daten is None:
        daten = datetime.now()
    return daten.strftime("%H:%M:%S")


def _date_str(daten=None):
    if daten is None:
        daten = datetime.now()
    return daten.strftime('%d-%b-%Y')


def _find(obj, name, default=None):
    if isinstance(name, str):
        if name in obj:
            return obj[name]
        else:
            return default
    elif name[0] in obj:
        if len(name) == 1:
            return obj[name[0]]
        else:
            return _find(obj[name[0]], name[1:], default)
    else:
        return default


def _trigger(snr, twin, threshold=0.0, ewin=0):
    trig_array = np.zeros(snr.shape, dtype='bool')
    trig = False
    mInd = 0
    mVal = snr[0]
    for ind, val in enumerate(snr):
        if trig:
            if ind > mInd+twin:
                trig = False
                trig_array[mInd] = True
                mInd = ind
                mVal = val
            elif val > mVal:
                mInd = ind
                mVal = val
        else:
            if (val > mVal) and (val > threshold):
                mInd = ind
                mVal = val
                trig = True
            else:
                mInd = ind
                mVal = val
    if (ewin > 0) and trig and (mInd + ewin < len(snr)):
        trig_array[mInd] = True
        trig = False
    det = trig_array.nonzero()
    tind = mInd if trig else None
    return det[0], tind



def _read_scan(fname):
    nbyte = path.getsize(fname)
    with open(fname, "rb") as fp:
        data = fp.read(32)
        hdr = np.fromstring(data, dtype='int64')
        nsamp = hdr[0]
        srate = hdr[2]
        # tinc  = (1000*1000*nsamp)/srate
        blk_size = 32 + 4*4*nsamp
        mxrec = int(nbyte / blk_size)
        dsnr = np.zeros((mxrec,nsamp),dtype='float32')
        dloc = np.zeros((mxrec,3*nsamp),dtype='float32')
        stime = np.zeros(mxrec,dtype='int64')
        for rec in range(mxrec):
            fp.seek(rec * blk_size)
            data = fp.read(32)
            hdr = np.fromstring(data, dtype='int64')
            stime[rec] = hdr[1]
            data = fp.read(4 * nsamp)
            dsnr[rec] = np.fromstring(data, dtype='float32')
            data = fp.read(12*nsamp)
            dloc[rec]  = np.fromstring(data, dtype='float32')



        #dsnr.resize((nrec,nsamp))
        #dloc.resize((nrec,nsamp*3))
        #stime.resize(nrec)

    return (dsnr, dloc, stime, srate)



class SeisTrigger01:
    '''
        Currently Takes an Outputted Scan File before triggering for an event.


    '''
    def __init__(self, window = [1.0, 10.0], threshold = [2.0, 0.8]):
        self.start_time = 0
        self.snr = np.array([],dtype='float32')
        self.loc = np.array([],dtype='float32')
        self.threshold = np.array(threshold)
        self.window = np.array(window)


    def process_results(self, file_name):
        dsnr, dloc, stime, srate = _read_scan(file_name)
        nrec = len(stime)
        dloc = dloc.reshape([nrec, -1, 3])

        etm = np.zeros((0,))
        esnr = np.zeros((0,))
        eloc = np.zeros((0, 3))
        for i in range(nrec):
            ttm, tsnr, tloc = self.write(dsnr[i, :], dloc[i, :, :], srate, stime[i])
            etm = np.r_[etm, ttm / 1000000]
            esnr = np.r_[esnr, tsnr]
            eloc = np.r_[eloc, tloc]
        return etm, esnr, eloc

    def write(self, snr, loc, srate, stime):
        threshold = np.array(self.threshold)
        window = (srate * np.array(self.window)).astype('int32')
        if len(snr) != len(loc):
            raise RuntimeError("SNR and LOCATION array are of different length.")
        if type(stime) != np.int64:
            stime = np.int64(1000*1000*float(stime))
        tinc = int(1000000/srate)
        next_time = self.start_time + len(self.snr)*tinc
        if abs(stime - next_time) > 1000:
            self.start_time = stime
            self.loc = loc
            self.snr = snr
        else:
            self.snr = np.concatenate([self.snr, snr])
            self.loc = np.concatenate([self.loc, loc])
        for twin, thresh in zip(window, threshold):
            det, trig = _trigger01(self.snr,twin,thresh)
            if len(det) > 1:
                break
        dsnr = self.snr[det]
        dloc = self.loc[det,:]
        dtm  = self.start_time + det*tinc
        if trig is None:
            self.start_time = 0
        else:
            #print('Trigger = ' + repr(trig))
            trig = trig - 1
            self.snr = self.snr[trig:]
            self.loc = self.loc[trig:,:]
            self.start_time = self.start_time + trig*tinc
        return dtm, dsnr, dloc


class SeisOutFile:
    '''
        Definition of manipulation types for the Seismic scan files.

    '''

    def __init__(self, path = '', name = None):
        self.open(path, name)

    def open(self, path = '', name = None):
        self.path = path
        if name is None:
            name = datetime.now().strftime('RUN_%Y%m%d_%H%M%S')
        self.name = name
        print('Path = ' + repr(self.path) + ', Name = ' + repr(self.name))

    def write_log(self, message):
        fname = path.join(self.path,self.name + '.log')
        with open(fname, "a") as fp:
            fp.write(message + '\n')

    def write_scan(self,dsnr,dloc,stime,srate):
        #str   = struct.pack('<q<l',int(1000*1000*float(daten),namp))
        fname = path.join(self.path,self.name + '.scn')
        with open(fname, "ab") as fp:
            nsamp = len(dsnr)
            hdr = np.zeros(4,dtype='int64')
            hdr[0] = nsamp
            hdr[1] = int(1000*1000*float(stime))
            hdr[2] = srate
            fp.write(hdr)
            fp.write(dsnr.astype('float32').tobytes())
            fp.write(dloc.astype('float32').tobytes())

    def read_scan(self):
        fname = path.join(self.path,self.name + '.scn')
        dsnr, dloc, stime, srate = _read_scan(fname)
        return (dsnr, dloc, stime, srate)

    def write_event(self, event_data, station_data):
        pass


class SeisScanParam:
    '''
       Class that reads in a user defined parameter file for all the required
    scanning Information

      _set_param - Definition of the path for the Parameter file to be read



    '''

    def __init__(self, param = None):
        self.lookup_table = None
        self.seis_reader = None
        self.bp_filter_p1 = [2.0, 16.0, 3]
        self.bp_filter_s1 = [2.0, 12.0, 3]
        self.onset_win_p1 = [0.2, 1.0]
        self.onset_win_s1 = [0.2, 1.0]
        self.station_p1 = None
        self.station_s1 = None
        self.detection_threshold = 3.0
        self.detection_downsample = 5
        self.detection_window = 3.0
        self.minimum_velocity = 3000.0
        self.marginal_window = [0.5, 3000.0]
        self.location_method = "Mean"
        self.time_step = 10
        if param:
            self.load(param)

    def _set_param(self, param):
        type = _find(param,("MODEL","Type"))
        if (type == "MATLAB_LKT"):
            path = _find(param,("MODEL","Path"))
            if path:
                decimate = _find(param,("MODEL","Decimate"))
                self.lookup_table = mload.lut02(path, decimate)

        type = _find(param,("SEISMIC","Type"))
        if (type == "DAT"):
            path = _find(param,("SEISMIC","Path"))
            if path:
                self.seis_reader = mload.dat_reader(path)

        if (type == 'MSEED'):
            path = _find(param,("SEISMIC","Path"))
            if path:
                self.seis_reader = mload.mseed_reader(path)

        #scn = _find(param,("SCAN"))
        #if scn:
        #    self.time_step = _find(scn,"TimeStep",self.time_step)
        #    path = _find(scn,"OutputPath")
        #    if path:
        #        self.output_stream.open(path)

        scn = _find(param,("PARAM"))
        if scn:
            self.time_step = _find(scn,"TimeStep",self.time_step)
            self.station_p1 = _find(scn,"StationSelectP",self.station_p1)
            self.station_s1 = _find(scn,"StationSelectS",self.station_s1)
            self.bp_filter_p1 = _find(scn,"SigFiltP1Hz",self.bp_filter_p1)
            self.bp_filter_s1 = _find(scn,"SigFiltS1Hz",self.bp_filter_s1)
            self.onset_win_p1 = _find(scn,"OnsetWinP1Sec",self.onset_win_p1)
            self.onset_win_s1 = _find(scn,"OnsetWinS1Sec",self.onset_win_s1)
            self.detection_downsample = _find(scn,"DetectionDownsample",self.detection_downsample)
            self.detection_window = _find(scn,"DetectionWindow",self.detection_window)
            self.minimum_velocity = _find(scn,"MinimumVelocity",self.minimum_velocity)
            self.marginal_window  = _find(scn,"MarginalWindow",self.marginal_window)
            self.location_method  = _find(scn,"LocationMethod",self.location_method)

    def _load_json(self, json_file):
        param = None
        with open(json_file,'r') as fp:
            param = json.load(fp)
        return param

    def load(self, file):
        param = self._load_json(file)
        self._set_param(param)


class SeisScan:

    def __init__(self, lut, reader=None, param=None, output_path=None, output_name=None):
        self.sample_rate = 1000.0
        self.seis_reader = reader
        self.lookup_table = lut

        if param is None:
            param = SeisScanParam()

        self.pre_pad = 0.0
        self.post_pad = 5.0
        self.time_step = 10.0

        self.bp_filter_p1 = param.bp_filter_p1
        self.bp_filter_s1 = param.bp_filter_s1
        self.onset_win_p1 = param.onset_win_p1
        self.onset_win_s1 = param.onset_win_s1
        self.boxcar_p1 = 0.050
        self.boxcar_s1 = 0.100
        self.station_p1 = param.station_p1
        self.station_s1 = param.station_s1
        self.detection_threshold = param.detection_threshold
        if output_path is not None:
            self.outputp1 = SeisOutFile(output_path, output_name + '_p1')
            self.outputs1 = SeisOutFile(output_path, output_name + '_s1')
            self.outputps = SeisOutFile(output_path, output_name + '_ps')
        else:
            self.outputp1 = None
            self.outputs1 = None
            self.outputps = None

        self.raw_data = dict()
        self.filt_data = dict()
        self.onset_data = dict()

        self._initialized = False
        self._station_name = None
        self._station_p1_flg = None
        self._station_s1_flg = None
        self._station_file = None
        self._map = None

    def _pre_proc_p1(self, sig_z, srate):
        lc, hc, ord = self.bp_filter_p1  # Apply a bandpass filter with information defined in ParameterFile/Inputs
        sig_z = filter(sig_z, srate, lc, hc, ord) # applies a butter filter
        return sig_z

    def _pre_proc_s1(self, sig_e, sig_n, srate):
        lc, hc, ord = self.bp_filter_s1  # Apply a bandpass filter with information defined in ParameterFile/Inputs
        sig_e = isp.filter(sig_e, srate, lc, hc, ord) # Applies a butter filter to E-component
        sig_n = isp.filter(sig_n, srate, lc, hc, ord) # Applies a butter filter to N-component
        return sig_e, sig_n

    def _compute_onset_p1(self, sig_z, srate):
        stw, ltw = self.onset_win_p1  # Define the STW and LTW for the onset function
        stw = int(stw * srate) + 1 # Changes the onset window to actual samples
        ltw = int(ltw * srate) + 1 # Changes the onset window to actual samples
        sig_z = self._pre_proc_p1(sig_z, srate) # Apply the pre-processing defintion
        self.filt_data['sigz'] = sig_z  # ???
        sig_z = onset(sig_z, stw, ltw) # Determine the onset function using definition
        self.onset_data['sigz'] = sig_z
        return sig_z

    def _compute_onset_s1(self, sig_e, sig_n, srate):
        stw, ltw = self.onset_win_s1
        stw = int(stw * srate) + 1
        ltw = int(ltw * srate) + 1
        sig_e, sig_n = self._pre_proc_s1(sig_e, sig_n, srate)
        self.filt_data['sige'] = sig_e
        self.filt_data['sign'] = sig_n
        sig_e = onset(sig_e, stw, ltw)
        sig_n = onset(sig_n, stw, ltw)
        self.onset_data['sige'] = sig_e
        self.onset_data['sign'] = sig_n
        snr = np.sqrt(sig_e * sig_e + sig_n * sig_n)
        self.onset_data['sigs'] = snr
        return snr

    def _compute_p1(self, stime, samples, stnp=None):
        srate = self.sample_rate

        sigz = samples[2]

        snr_p1 = self._compute_onset_p1(sigz, srate) # Computes the onset value from the data

        if (self.boxcar_p1 > 0.0):
            bc_winp = np.rint(self.boxcar_p1 * srate)
            snr_p1 = ilib.boxcar_filt(snr_p1, bc_winp)

        ttp = self.lookup_table.fetch_index('TIME_P1', srate)

        nchan, tsamp = snr_p1.shape # Defining the number of signals and the total time samp;e

        pre_smp = int(self.pre_pad * srate) # Defining the pre-pading as sample number
        pos_smp = int(self.post_pad * srate) # Defining the post-pad as sample number

        nsamp = tsamp - pre_smp - pos_smp # Defining the total length to scan
        daten = stime - pre_smp / srate # Defining the initial datum to take samples from

        ncell = tuple(self.lookup_table.cell_count) # Defining the size of the structure to load


        # Generating the full Coalescence grid that will be sequentially filled. This needs to be altered to  not generate all in memory.
        #Instead pass the size into the C-code and sequentially save to file instead of variable. NEED TO RE-WRITE THE C-Code.

        if self._map is None:
            print('  Allocating memory: {}'.format(ncell + (tsamp,)))
            self._map = np.zeros(ncell + (tsamp,), dtype=np.float64)


        dind = np.zeros(tsamp, np.int64)
        dsnr = np.zeros(tsamp, np.double)

        ilib.scan(snr_p1, ttp, pre_smp, pre_smp + nsamp, self._map) # This takes all the required data and scans the data accourdingly ~ POSSIBLE ADDITION OF SEGMENTING TABLE
        ilib.detect(self._map, dsnr, dind, pre_smp, pre_smp + nsamp) # This then detects from that data
        dsnr_p = np.exp((dsnr[pre_smp:pre_smp + nsamp] / (nchan/2)) - 1.0)
        dloc_p = self.lookup_table.index2xyz(dind[pre_smp:pre_smp + nsamp])

        return daten, dsnr_p, dloc_p

    def _compute_s1(self, stime, samples, stnp=None):
        srate = self.sample_rate

        sigx = samples[0]
        sigy = samples[1]

        snr_s1 = self._compute_onset_s1(sigx, sigy, srate)

        if (self.boxcar_s1 > 0.0):
            bc_wins = np.rint(self.boxcar_s1 * srate)
            snr_s1 = ilib.boxcar_filt(snr_s1, bc_wins)

        tts = self.lookup_table.fetch_index('TIME_S1', srate)

        nchan, tsamp = snr_s1.shape

        pre_smp = int(self.pre_pad * srate)
        pos_smp = int(self.post_pad * srate)

        nsamp = tsamp - pre_smp - pos_smp
        daten = stime - pre_smp / srate

        ncell = tuple(self.lookup_table.cell_count)

        if self._map is None:
            print('  Allocating memory: {}'.format(ncell + (tsamp,)))
            self._map = np.zeros(ncell + (tsamp,), dtype=np.float64)

        dind = np.zeros(tsamp, np.int64)
        dsnr = np.zeros(tsamp, np.double)

        ilib.scan01(snr_s1, tts, pre_smp, pre_smp + nsamp, self._map)
        ilib.detect01(self._map, dsnr, dind, pre_smp, pre_smp + nsamp)
        dsnr_s = np.exp((dsnr[pre_smp:pre_smp + nsamp] / (nchan/2)) - 1.0)
        dloc_s = self.lookup_table.index2xyz(dind[pre_smp:pre_smp + nsamp])

        return daten, dsnr_s, dloc_s

    def _compute_ps(self, stime, samples, stnp=None):
        srate = self.sample_rate

        sige = samples[0]
        sign = samples[1]
        sigz = samples[2]

        snr_p1 = self._compute_onset_p1(sigz, srate)
        snr_s1 = self._compute_onset_s1(sige, sign, srate)

        if (self.boxcar_p1 > 0.0):
            bc_winp = np.rint(self.boxcar_p1 * srate)
            snr_p1 = ilib.boxcar_filt(snr_p1, bc_winp)
        if self.boxcar_s1 > 0.0:
            bc_wins = np.rint(self.boxcar_s1 * srate)
            snr_s1 = ilib.boxcar_filt(snr_s1, bc_wins)

        snr = np.concatenate((snr_p1, snr_s1))

        ttp = self.lookup_table.fetch_index('TIME_P1', srate)
        tts = self.lookup_table.fetch_index('TIME_S1', srate)

        tt = np.c_[ttp, tts]

        nchan, tsamp = snr.shape

        pre_smp = int(self.pre_pad * srate)
        pos_smp = int(self.post_pad * srate)

        nsamp = tsamp - pre_smp - pos_smp
        daten = stime - pre_smp / srate

        ncell = tuple(self.lookup_table.cell_count)

        if self._map is None:
            print('  Allocating memory: {}'.format(ncell + (tsamp,)))
            self._map = np.zeros(ncell + (tsamp,), dtype=np.float64)

        dind = np.zeros(tsamp, np.int64)
        dsnr = np.zeros(tsamp, np.double)

        ilib.scan(snr, tt, pre_smp, pre_smp + nsamp, self._map)
        ilib.detect(self._map, dsnr, dind, pre_smp, pre_smp + nsamp)
        dsnr_ps = np.exp((dsnr[pre_smp:pre_smp + nsamp] / nchan) - 1.0)
        dloc_ps = self.lookup_table.index2xyz(dind[pre_smp:pre_smp + nsamp])

        return daten, dsnr_ps, dloc_ps

    def _compute(self, stime, samples, stnp=None, stns=None):
        srate = self.sample_rate

        sige = samples[0]
        sign = samples[1]
        sigz = samples[2]

        snr_p1 = self._compute_onset_p1(sigz, srate)
        snr_s1 = self._compute_onset_s1(sige, sign, srate)

        if (self.boxcar_p1 > 0.0):
            bc_winp = np.rint(self.boxcar_p1 * srate)
            snr_p1 = ilib.boxcar_filt(snr_p1, bc_winp)
        if self.boxcar_s1 > 0.0:
            bc_wins = np.rint(self.boxcar_s1 * srate)
            snr_s1 = ilib.boxcar_filt(snr_s1, bc_wins)

        snr = np.concatenate((snr_p1, snr_s1))

        ttp = self.lookup_table.fetch_index('TIME_P1', srate)
        tts = self.lookup_table.fetch_index('TIME_S1', srate)

        tt = np.c_[ttp, tts]

        nchan, tsamp = snr.shape

        pre_smp = int(self.pre_pad * srate)
        pos_smp = int(self.post_pad * srate)

        nsamp = tsamp - pre_smp - pos_smp
        daten = stime - pre_smp / srate

        ncell = tuple(self.lookup_table.cell_count)

        if self._map is None:
            print('  Allocating memory: {}'.format(ncell + (tsamp,)))
            self._map = np.zeros(ncell + (tsamp,), dtype=np.float64)

        dind = np.zeros(tsamp, np.int64)
        dsnr = np.zeros(tsamp, np.double)

        # To understand how I can segment the code I need to fully understand the inputs for the functions.
        #    ~ snr_p1   - Onset function for Z-component
        #    ~ ttp      - Travel-time of P-Wave
        #    ~ pre_smp  - The padding for the start of the data
        #    ~  _map    - The large blank 4D coalescence grid
        ilib.scan(snr_p1, ttp, pre_smp, pre_smp + nsamp, self._map)
        ilib.detect(self._map, dsnr, dind, pre_smp, pre_smp + nsamp)
        dsnr_p = np.exp((dsnr[pre_smp:pre_smp + nsamp] / (nchan/2)) - 1.0)
        dloc_p = self.lookup_table.index2xyz(dind[pre_smp:pre_smp + nsamp])



        ilib.scan(snr_s1, tts, pre_smp, pre_smp + nsamp, self._map)
        ilib.detect(self._map, dsnr, dind, pre_smp, pre_smp + nsamp)
        dsnr_s = np.exp((dsnr[pre_smp:pre_smp + nsamp] / (nchan/2)) - 1.0)
        dloc_s = self.lookup_table.index2xyz(dind[pre_smp:pre_smp + nsamp])

        ilib.scan(snr, tt, pre_smp, pre_smp + nsamp, self._map)
        ilib.detect(self._map, dsnr, dind, pre_smp, pre_smp + nsamp)
        dsnr_ps = np.exp((dsnr[pre_smp:pre_smp + nsamp] / nchan) - 1.0)
        dloc_ps = self.lookup_table.index2xyz(dind[pre_smp:pre_smp + nsamp])

        return daten, dsnr_p, dloc_p, dsnr_s, dloc_s, dsnr_ps, dloc_ps

    def compute(self, stime, samples, output_path=None, output_name=None):

        if output_path is not None:
            self.outputp1 = SeisOutFile(output_path, output_name + '_p1')
            self.outputs1 = SeisOutFile(output_path, output_name + '_s1')
            self.outputps = SeisOutFile(output_path, output_name + '_ps')

        print('=====================================================')
        print('   SCAN: {} - {}'.format(self.outputp1.path, self.outputp1.name))
        print('=====================================================')

        nsamp = samples.shape[-1]

        srate = self.sample_rate
        step = int(self.time_step*srate+0.5)
        pre_smp = int(self.pre_pad * srate)
        pos_smp = int(self.post_pad * srate)
        wsamp = step + pre_smp + pos_smp
        fsamp = np.arange(0, nsamp-wsamp+1, step)
        cnt = 0
        for fs in fsamp:
            cnt += 1
            ftime = stime + fs/srate
            tstr = datetime.now().strftime('%H:%M:%S')
            print('{}: Scanning {}/{}'.format(tstr, cnt, fsamp.size))
            print('  TIME {}, Sample rage {} - {}'.format(ftime, fs, fs+wsamp))
            sig = samples[:, :, fs:fs+wsamp]
            daten, dsnr_p1, dloc_p1, dsnr_s1, dloc_s1, dsnr_ps, dloc_ps = self._compute(ftime, sig)
            self.outputp1.write_scan(dsnr_p1, dloc_p1, daten, srate)
            self.outputs1.write_scan(dsnr_s1, dloc_s1, daten, srate)
            self.outputps.write_scan(dsnr_ps, dloc_ps, daten, srate)




