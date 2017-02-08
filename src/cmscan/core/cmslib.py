
import os
import numpy as np
import numpy.ctypeslib as clib

c_int = clib.ctypes.c_int
c_int8 = clib.ctypes.c_int8
c_int16 = clib.ctypes.c_int16
c_int32 = clib.ctypes.c_int32
c_int64 = clib.ctypes.c_int64
c_dbl = clib.ctypes.c_double
c_dPt = clib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_dPt = clib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_i8Pt = clib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
c_i16Pt = clib.ndpointer(dtype=np.int16, flags="C_CONTIGUOUS")
c_i32Pt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
c_i64Pt = clib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
c_iPt = clib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

if os.name == 'nt':
    _cmslib = clib.load_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/cmslib.dll'), '.')
else:  # posix
    _cmslib = clib.load_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/cmslib.so'), '.')

_cmslib.onset.argtypes = [c_dPt, c_int, c_int, c_int, c_int, c_dPt]
_cmslib.onset_mp.argtypes = [c_dPt, c_int, c_int, c_int, c_int, c_int, c_dPt]

def onset(env, stw, ltw, gap):
    ntr = env[..., 0].size
    nsamp = env.shape[-1]
    out = np.zeros(env.shape, dtype=np.float64)
    env = np.ascontiguousarray(env, np.float64)
    if ntr > 1:
        _cmslib.onset_mp(env, ntr, nsamp, int(stw), int(ltw), int(gap), out)
        return out
    else:
        _cmslib.onset(env, nsamp, int(stw), int(ltw), int(gap), out)
        return out