"""
Various functions used with the evclass module

Gunnar Eggertsson, March 2024.
"""

# Standard library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz
import pickle
from obspy import UTCDateTime
from obspy.core import stream, trace
from obspy.clients.fdsn import Client
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# Local application imports
from silio import ReadGrx, read_eve, read_evlib, lib2grx, write_grx
from SNSNPy.io import swd_client, smd_client
from SNSNPy import snsn_tt
from SNSNPy.sql_db import combull, sil
from seis import LocationDist


############################################################# VARIABLES #################################################################################

# Paths on fenrir
BASE_PATH            = "/mnt/sil/eq1"                               # Base path to eq1 archive
NAS_PATH             = "/mnt/snsn_data/eq1"                         # Base path to NAS
sta_path             = "/home/gunnar/evclass/net.dat"               # Path to stations file
evclass_path         = "/home/gunnar/python"                        # Path to the evclass scripts
evclass_models_path  = "/home/gunnar/python/models/2022"            # Base path to evclass models
evornot_models_path  = "/home/gunnar/python/models/evornot"         # Base path to evornot models
wfdata_path          = "/home/gunnar/evclass/waveformdata/psreq"    # Path to waveform data (NumPy arrays after data pre-preocessing)
grx_path             = "/home/gunnar/evclass/grx/psreq"             # Path to grx files
velmod_path          = "/home/gunnar/evclass"                       # Path to velocity model files

"""
# Paths on seppl
BASE_PATH            = "/mnt/sil/eq1"
NAS_PATH             = "/mnt/snsn_data/eq1"                                 
sta_path             = "/home/snsn/silbull/etc/net.dat"
evclass_path         = "/home/snsn/silbull"
evclass_models_path  = "/home/snsn/silbull/models/2022"
evornot_models_path  = "/home/snsn/silbull/models/evornot"
wfdata_path          = "/home/snsn/silbull/wf_data"
velmod_path          = "/home/snsn/silbull/etc"
grx_path             = "/home/snsn/silbull/grx"
"""

# Paths on yggdrasil
"""
BASE_PATH            = "/eq1"                                               # Base path to eq1 archive
sta_path             = "/home/sil/python/etc/net.dat"
evclass_models_path  = "/home/sil/python/models/2022"
evornot_models_path  = "/home/sil/python/models/evornot"
velmod_path          = "/home/sil/etc"
"""

# Paths on slink6
"""
sta_path             = "/home/sc/evclass/etc/net.dat"
evclass_models_path  = "/home/sc/evclass/models/2022"
evornot_models_path  = "/home/sc/evclass/models/evornot"
velmod_path          = "/home/sc/evclass/etc"
"""

f_lo               = np.insert(np.arange(2, 39, 2), 0, 1)   # Lower end of pass-band
n_channels         = 3                                      # Number of channels
n_tws              = 4                                      # Number of time windows for each channel (4: P, Pc, S, Sc)
d2r                = np.pi / 180.0                          # Degrees to radians
dist_lim_mines_sil = 25                                     # If an event is more than this distance (in km) away from Kiru/Malm, it is not taken to be in the mine
dist_lim_mines_com = 25

stas = [
    "aal",
    "arn",
    "ask",
    "bac",
    "bju",
    "ble",
    "bog",
    "bor",
    "bre",
    "bur",
    "byx",
    "del",
    "dun",
    "eks",
    "ert",
    "esk",
    "fab",
    "fal",
    "fib",
    "fin",
    "fkp",
    "fla",
    "fly",
    "for",
    "gno",
    "got",
    "gra",
    "har",
    "has",
    "hem",
    "hot",
    "hus",
    "igg",
    "kal",
    "kov",
    "kur",
    "lan",
    "lil",
    "lnk",
    "lun",
    "mas",
    "nas",
    "nik",
    "nor",
    "nra",
    "nrt",
    "nyn",
    "ode",
    "ona",
    "osk",
    "ost",
    "paj",
    "rat",
    "rot",
    "sal",
    "sju",
    "sol",
    "sto",
    "str",
    "sva",
    "tjo",
    "udd",
    "uma",
    "up1",
    "van",
    "vik",
    "vst",
    "vxj",
]

stas_4cl = ["kur", "rat", "kov", "nik", "dun", "mas"]
stas_decom = ["udd", "uma"]
stas_old = ["asp", "hud", "ons"]

# Temporary Hälsingland network
stas_hals = [
    "fag",
    "mor",
    "hog",
    "sod",
    "bas",
    "hal",
    "sjv",
    "gad",
    "fus",
    "ess",
    "orb",
    "lus",
    "dju",
]

stas_excl = ["VBYGD", "VAGH", "OSL", "COP", "MUD", "KONS", "ROEST", "JETT"]

typs_allowed = ["ex", "qu", "ql", "me"]

type_conv = {"ex": 0, "qu": 1, "ql": 2, "me": 3}

type_conv_rev = {"0": "ex", "1": "qu", "2": "ql", "3": "me"}

type_conv_rev_mines = {"0": "ex", "1": "ql", "2": "me"}


# Columns for the event classifier (waveform features)
tws = [
    "Z-P",
    "Z-Pc",
    "Z-S",
    "Z-Sc",
    "R-P",
    "R-Pc",
    "R-S",
    "R-Sc",
    "T-P",
    "T-Pc",
    "T-S",
    "T-Sc",
]
cols = []
for i in np.insert(np.arange(2, 39, 2), 0, 1):
    cols = cols + [str(i) + "-" + x for x in tws]
cols.append("typ")

mines = {
    "kiru": [67.8322, 20.1782],
    "mert": [67.7095, 20.7917],
    "svap": [67.6309, 20.9884],
    "malm": [67.1885, 20.6992],
    "aiti": [67.0719, 20.9532],
    "rens": [64.9234, 20.0957],
    "kank": [64.9118, 20.2408],
    "bjor": [64.9360, 20.5692],
}

sta_typs_allowed = {}
for sta in stas:
    if sta in stas_4cl:
        sta_typs_allowed[sta] = typs_allowed
    else:
        sta_typs_allowed[sta] = typs_allowed[:-2]


def generate_waveformdata(
    arrT_p, arrT_s, staU, network, ev_lat, ev_lon, plot=False, lf=2.0, hf=20.0
):

    f_lo      = np.insert(np.arange(2, 39, 2), 0, 1)            # Lower end of pass-band
    pre_time  = 60                                              # Number of seconds to fetch before P-arrival
    dur       = 180                                             # Number of seconds to fetch after start time
    startT    = arrT_p - datetime.timedelta(seconds=pre_time)   # Start time to fetch
    endT      = startT + datetime.timedelta(seconds=dur)        # End time to fetch
    ev_coord  = [ev_lat, ev_lon]                                # Event location (for rotation of horizontals)
    diff_time = arrT_s - arrT_p                                 # Differential travel time

    """
    Plots for the paper
    pre_time = 2
    dur = 30
    """

    ########################################################################################################################
    # Get data from the dataserver or FDSN server

    # Dataserver
    FDSN = False
    try:
        if network not in ["UP", "MS"]:
            raise ValueError

        getT  = int(startT - UTCDateTime(0))                            # Start time in unix time (for the dataserver)
        srate = 100                                                     # Sampling rate
        days_diff = int((datetime.date.today() - arrT_p.date).days)
        if days_diff <= 7:
            (D, n) = swd_client.SwdData(
                network, staU, getT, dur + 1, server=["130.238.140.21", 48673, 1]
            )
        else:
            (D, n) = swd_client.SwdData(
                network, staU, getT, dur + 1, server=["130.238.140.20", 48673, 1]
            )

    # FDSN server
    except Exception as e:
        print("Failed to fetch data from dataserver. Falling back to FDSN server.", e)
        try:
            fdsn = "http://130.238.140.89:8080"
            t1_s = "%4d-%02d-%02dT%02d:%02d:%06.3f" % (
                startT.year,
                startT.month,
                startT.day,
                startT.hour,
                startT.minute,
                startT.second,
            )
            t1 = UTCDateTime(t1_s)
            t2_s = "%4d-%02d-%02dT%02d:%02d:%06.3f" % (
                endT.year,
                endT.month,
                endT.day,
                endT.hour,
                endT.minute,
                endT.second,
            )
            t2 = UTCDateTime(t2_s)
            client = Client(fdsn)
            if network in ["UP", "MS"]:
                srate = 100
                stz = client.get_waveforms(
                    network=network,
                    station=staU,
                    location="*",
                    channel="HH*",
                    starttime=t1,
                    endtime=t2,
                )
                stz.sort()  # ENZ
                D = [
                    [float(i) for i in list(stz[2].data)],
                    [float(i) for i in list(stz[1].data)],
                    [float(i) for i in list(stz[0].data)],
                ]

            else:
                try:
                    stz = client.get_waveforms(
                        network=network,
                        station=staU,
                        location="*",
                        channel="HH*",
                        starttime=t1,
                        endtime=t2,
                        attach_response=True,
                    )
                    ch = "HHZ"
                except Exception as e:
                    stz = client.get_waveforms(
                        network=network,
                        station=staU,
                        location="*",
                        channel="CH*",
                        starttime=t1,
                        endtime=t2,
                        attach_response=True,
                    )
                    ch = "CHZ"

                srate = int(stz[0].stats.sampling_rate)

                # Identify FIRResponse stages
                inv_res = client.get_stations(
                    network=network,
                    station=staU,
                    channel=ch,
                    starttime=t1,
                    endtime=t2,
                    level="response",
                )
                loc = inv_res[0][0][0].location_code
                if loc:
                    seed_id = (
                        network
                        + "."
                        + inv_res[0][0].code
                        + "."
                        + loc
                        + "."
                        + inv_res[0][0][0].code
                    )
                else:
                    seed_id = (
                        network
                        + "."
                        + inv_res[0][0].code
                        + ".."
                        + inv_res[0][0][0].code
                    )
                response = inv_res.get_response(seed_id, t1)
                FIR_stage_found = False
                for stage in response.response_stages:
                    # print(stage.__class__.__name__)
                    if FIR_stage_found:
                        if "FIR" in stage.__class__.__name__:
                            continue
                        else:
                            print(staU, " -- Different stage after FIR...")
                    # print(stage.__class__.__name__)
                    # See https://docs.obspy.org/packages/autogen/obspy.core.inventory.response.ResponseStage.html#obspy.core.inventory.response.ResponseStage
                    if "FIR" in stage.__class__.__name__:
                        FIR_stage_found = True
                        end_stage = stage.stage_sequence_number - 1

                pfilt = (0.02, 0.033, 44.0, 48.0)
                stz.detrend(type="linear")
                if FIR_stage_found:
                    stz.remove_response(
                        output="vel",
                        pre_filt=pfilt,
                        zero_mean=True,
                        taper=True,
                        taper_fraction=0.05,
                        end_stage=end_stage,
                    )
                else:
                    stz.remove_response(
                        output="vel",
                        pre_filt=pfilt,
                        zero_mean=True,
                        taper=True,
                        taper_fraction=0.05,
                    )

                if srate != int(stz[0].stats.sampling_rate):
                    print(
                        staU,
                        " -- Sampling rate changed after removal of instrument response",
                    )

                inv = client.get_stations(
                    network=network,
                    station=staU,
                    starttime=t1,
                    endtime=t2,
                    level="channel",
                )
                sta_lat = inv[0][0].latitude
                sta_lon = inv[0][0].longitude
                sta_coord = [sta_lat * d2r, sta_lon * d2r]
                D = LocationDist(2, ev_coord, sta_coord, -1)
                baz = D[2] / d2r
                stz.rotate(method="NE->RT", back_azimuth=baz)
                for tr in stz:
                    tr.data *= 1e9

                stz.sort()  # RTZ
                D = [
                    [float(i) for i in list(stz[2].data)],
                    [float(i) for i in list(stz[0].data)],
                    [float(i) for i in list(stz[1].data)],
                ]

            FDSN = True
        except Exception as e:
            print("Failed to fetch data from FDSN server", e)
            return float("NaN")

    if srate < 80:
        return float("NaN")

    """
    We delete the first k entries of the data vectors so that the first entry occurs exactly <pre_time> seconds
    before the P-phase arrival. This way the picked P-phase should always occur exactly 10 seconds after the first data point.

    What I think happens:
    The P-arrival time is fetched from the eve file, e.g.
    p_arr      = 2021-01-02T09:23:49.760000 - then we have (if pre_time = 10s):
    startT = 2021-01-02T09:23:39.760000 
    then the above command, instead of fetching data from 23.39.760000 will fetch data from 23.39.000000,
    i.e. floor down to a whole second (and then also end at XX.XX.000000). The variable k finds the index of the actual starting
    point we want (23.49.760000) and then we delete all entries up to that point. 
    The result is data that starts exactly 10 seconds before the P-arrival but, consequently will have length <dur>-k rather than <dur>
    To make up for this I add one extra second and cut off the end to make the data exactly <dur> seconds long.
    """
    k = int(np.round(startT.microsecond / 10000.0))  # Number of data points to delete
    K = int(k / 100 * srate)
    for i in range(3):
        del D[i][0:K]
        del D[i][-1 - srate + K : -1]

    if FDSN:
        for i in range(3):
            del D[i][-2:]
    else:
        for i in range(3):
            del D[i][-1:]

    st = stream.Stream()
    stats = trace.Stats()  # Insert metadata
    stats.sampling_rate = srate
    stats.npts = len(D[0])
    stats.starttime = startT
    stats.network = network
    stats.station = staU

    # Remove instrument response
    if network in ["UP", "MS"]:
        stats.channel = "Z"
        z = trace.Trace(np.array(D[0]), stats)
        stats.channel = "N"
        n = trace.Trace(np.array(D[1]), stats)
        stats.channel = "E"
        e = trace.Trace(np.array(D[2]), stats)

        st += stream.Stream(traces=[n, z, e])
        st = remove_IR(st)
        st = rotate_zrt(st, ev_coord)

    else:
        stats.channel = "Z"
        z = trace.Trace(np.array(D[0]), stats)
        stats.channel = "R"
        r = trace.Trace(np.array(D[1]), stats)
        stats.channel = "T"
        t = trace.Trace(np.array(D[2]), stats)

        st += stream.Stream(traces=[r, z, t])

    ########################################################################################################################
    # Generate the classification data, RMS amplitudes in different time windows, filtered in different frequency bands.

    slices_evornot = {
        "start_ind": 0,
        "p_ind": int((pre_time - 1 * diff_time) * srate),
        "pc_ind": int((pre_time + 0 * diff_time) * srate),
        "s_ind": int((pre_time + 1 * diff_time) * srate),
        "sc_ind": int((pre_time + 2 * diff_time) * srate),
        "coda_ind": int((pre_time + 3 * diff_time) * srate),
        "final_ind": -1,
    }

    slices_evclass = {
        "start_ind": 0,
        "p_ind": int((pre_time + 0 * diff_time / 2.0) * srate),
        "pc_ind": int((pre_time + 1 * diff_time / 2.0) * srate),
        "s_ind": int((pre_time + 2 * diff_time / 2.0) * srate),
        "sc_ind": int((pre_time + 3 * diff_time / 2.0) * srate),
        "coda_ind": int((pre_time + 4 * diff_time / 2.0) * srate),
        "final_ind": -1,
    }

    n_feats = n_channels * n_tws
    evornot_data = np.empty(len(f_lo) * n_feats)
    evclass_data = np.empty(len(f_lo) * n_feats)

    if plot:
        plot_traces(st, slices_evclass, staU, startT, lf=lf, hf=hf)

    if srate == 80:
        f_lo[-1] = 36

    for j in range(len(f_lo)):
        st_copy = st.copy()
        if f_lo[j] == 1:
            f_hi = f_lo[j] + 2
        else:
            f_hi = f_lo[j] + 3

        st_copy.filter("bandpass", freqmin=f_lo[j], freqmax=f_hi) 

        Est_N, Est_P, Est_Pc, Est_S, Est_Sc, Est_C = extract_windows(st_copy, slices_evornot)
        st_N, st_P, st_Pc, st_S, st_Sc, st_C       = extract_windows(st_copy, slices_evclass)

        # Event-or-not
        evornot_data[j * n_feats + 0]  = rms(Est_P["Z"])
        evornot_data[j * n_feats + 1]  = rms(Est_Pc["Z"])
        evornot_data[j * n_feats + 2]  = rms(Est_S["Z"])
        evornot_data[j * n_feats + 3]  = rms(Est_Sc["Z"])
        evornot_data[j * n_feats + 4]  = rms(Est_P["R"])
        evornot_data[j * n_feats + 5]  = rms(Est_Pc["R"])
        evornot_data[j * n_feats + 6]  = rms(Est_S["R"])
        evornot_data[j * n_feats + 7]  = rms(Est_Sc["R"])
        evornot_data[j * n_feats + 8]  = rms(Est_P["T"])
        evornot_data[j * n_feats + 9]  = rms(Est_Pc["T"])
        evornot_data[j * n_feats + 10] = rms(Est_S["T"])
        evornot_data[j * n_feats + 11] = rms(Est_Sc["T"])

        # Evclass
        evclass_data[j * n_feats + 0]  = rms(st_P["Z"])
        evclass_data[j * n_feats + 1]  = rms(st_Pc["Z"])
        evclass_data[j * n_feats + 2]  = rms(st_S["Z"])
        evclass_data[j * n_feats + 3]  = rms(st_Sc["Z"])
        evclass_data[j * n_feats + 4]  = rms(st_P["R"])
        evclass_data[j * n_feats + 5]  = rms(st_Pc["R"])
        evclass_data[j * n_feats + 6]  = rms(st_S["R"])
        evclass_data[j * n_feats + 7]  = rms(st_Sc["R"])
        evclass_data[j * n_feats + 8]  = rms(st_P["T"])
        evclass_data[j * n_feats + 9]  = rms(st_Pc["T"])
        evclass_data[j * n_feats + 10] = rms(st_S["T"])
        evclass_data[j * n_feats + 11] = rms(st_Sc["T"])

        # print(" Filter band %02d (%02d - %02d Hz): PZ: %7.3f ; PR: %7.3f ; PT: %7.3f ; SZ: %7.3f ; SR: %7.3f ; ST: %7.3f"
        #    %(j+1, f_lo[j], f_hi, np.mean(np.abs(st_P["Z"].data)), np.mean(np.abs(st_P["R"].data)), np.mean(np.abs(st_P["T"].data)),
        #        np.mean(np.abs(st_S["Z"].data)), np.mean(np.abs(st_S["R"].data)), np.mean(np.abs(st_S["T"].data))))

    return (evornot_data, evclass_data)


def remove_IR(st):   
    for s in st:  # Use the metadata server
        sta = s.stats.station
        net = s.stats.network
        meta, err = smd_client.load_smd(
            station=sta, t=s.stats.starttime, network=net, cmplx_ret=True
        )
        if err:
            print("remove_IR", err, sta, s.stats.starttime)
            return float("NaN")

        # Convert to paz for removal
        A0 = meta[net][sta][1][-1][0]
        totSens = meta[net][sta][1][-1][1]
        ps = meta[net][sta][1][-1][2]
        zs = meta[net][sta][1][-1][3]
        paz = {"gain": A0, "sensitivity": totSens, "poles": ps, "zeros": zs}
        s.detrend(type="linear")  # Remoive the instrument response
        s.taper(type="hann", max_percentage=0.005)
        pfilt = (0.02, 0.033, 44.0, 48.0)  # 30s
        s.simulate(
            paz_remove=paz,
            paz_simulate=None,
            pre_filt=pfilt,
            zero_mean=True,
            taper=True,
            taper_fraction=0.05,
            pitsasim=False,
            sacsim=True,
        )

        s.data *= 1e9  # Convert to nm/s

    return st


def rotate_zrt(st, ev_coord):
    # Get the back azimuth from meta data server
    for s in st:
        sta = s.stats.station
        net = s.stats.network
        meta, err = smd_client.load_smd(station=sta, network=net)
        if err:
            print("rotate_zrt", err, sta, s.stats.starttime)
            return float("NaN")

        sta_coord = [meta[net][sta][0][1] * d2r, meta[net][sta][0][2] * d2r, 0.0]
        D = LocationDist(2, ev_coord, sta_coord, -1)
        s.stats.back_azimuth = D[2] / d2r
        # print(s.stats.back_azimuth)
        # print("%s baz %.2f" % (s.stats.station,D[2]/d2r))
    st.detrend(type="linear")
    st.rotate("NE->RT")
    return st


def rms(arr):
    return np.sqrt(np.mean(arr**2))


def extract_windows(st, slices):
    n_channels = len(st)
    st_N = {}
    st_P = {}
    st_Pc = {}
    st_S = {}
    st_Sc = {}
    st_C = {}
    for i in range(n_channels):
        cha = st[i].stats.channel                                       # Get the channel ("Z", "R", "T")
        st_N[cha]  = st[i][slices["start_ind"] : slices["p_ind"]]       # Noise window
        st_P[cha]  = st[i][slices["p_ind"]     : slices["pc_ind"]]      # P window
        st_Pc[cha] = st[i][slices["pc_ind"]    : slices["s_ind"]]       # P coda window
        st_S[cha]  = st[i][slices["s_ind"]     : slices["sc_ind"]]      # S window
        st_Sc[cha] = st[i][slices["sc_ind"]    : slices["coda_ind"]]    # S coda window
        st_C[cha]  = st[i][slices["coda_ind"]  : slices["final_ind"]]   # Coda window

    return (st_N, st_P, st_Pc, st_S, st_Sc, st_C)


def plot_traces(st, slices, sta, startT, lf=2.0, hf=20.0):
    times = st[0].times()  # Extract the times
    # times = [x-60 for x in times]
    st_copy = st.copy()
    st_copy.filter(
        "bandpass", freqmin=lf, freqmax=hf
    )  # Filter in different frequency bands
    stc_N, stc_P, stc_Pc, stc_S, stc_Sc, stc_C = extract_windows(
        st_copy, slices
    )  # Extract the filtered windows

    try:
        max_val = max(
            [
                max(stc_N["Z"][100:]),
                max(stc_P["Z"][100:]),
                max(stc_Pc["Z"][100:]),
                max(stc_S["Z"][100:]),
                max(stc_Sc["Z"][100:]),
                max(stc_C["Z"][100:]),
                max(stc_N["R"][100:]),
                max(stc_P["R"][100:]),
                max(stc_Pc["R"][100:]),
                max(stc_S["R"][100:]),
                max(stc_Sc["R"][100:]),
                max(stc_C["R"][100:]),
                max(stc_N["T"][100:]),
                max(stc_P["T"][100:]),
                max(stc_Pc["T"][100:]),
                max(stc_S["T"][100:]),
                max(stc_Sc["T"][100:]),
                max(stc_C["T"][100:]),
            ]
        )

        min_val = min(
            [
                min(stc_N["Z"][100:]),
                min(stc_P["Z"][100:]),
                min(stc_Pc["Z"][100:]),
                min(stc_S["Z"][100:]),
                min(stc_Sc["Z"][100:]),
                min(stc_C["Z"][100:]),
                min(stc_N["R"][100:]),
                min(stc_P["R"][100:]),
                min(stc_Pc["R"][100:]),
                min(stc_S["R"][100:]),
                min(stc_Sc["R"][100:]),
                min(stc_C["R"][100:]),
                min(stc_N["T"][100:]),
                min(stc_P["T"][100:]),
                min(stc_Pc["T"][100:]),
                min(stc_S["T"][100:]),
                min(stc_Sc["T"][100:]),
                min(stc_C["T"][100:]),
            ]
        )
        minmax = False
    except Exception as e:
        print("Failed to compute stream amplitude range", e)
        minmax = False

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fs = 25

    label_size = 16
    ax3.xaxis.set_tick_params(labelsize=label_size)
    ax1.yaxis.set_tick_params(labelsize=label_size)
    ax2.yaxis.set_tick_params(labelsize=label_size)
    ax3.yaxis.set_tick_params(labelsize=label_size)

    colp = "r"
    cols = "b"
    colc = "b"

    pick_height = 1.0
    ax1.set_title(
        "Station %s. Filtered %.1f - %.1f Hz" % (sta.upper(), lf, hf), fontsize=25
    )
    ax1.plot(
        times[:-1],
        np.concatenate(
            (stc_N["Z"], stc_P["Z"], stc_Pc["Z"], stc_S["Z"], stc_Sc["Z"], stc_C["Z"])
        ),
        "k",
    )
    if minmax:
        ax1.set_ylim([min_val, max_val])
    ymin, ymax = ax1.get_ylim()
    ax1.vlines(
        times[slices["p_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colp,
        linewidth=2,
        linestyle="dashed",
    )
    ax1.annotate(
        "P",
        ((times[slices["p_ind"]] + times[slices["pc_ind"]]) / 2, ymax * 0.8),
        color="r",
        fontsize=fs,
    )
    ax1.vlines(
        times[slices["pc_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colp,
        linewidth=2,
        linestyle="dashed",
    )
    ax1.annotate(
        "Pc",
        ((times[slices["pc_ind"]] + times[slices["s_ind"]]) / 2, ymax * 0.8),
        color="r",
        fontsize=fs,
    )
    ax1.vlines(
        times[slices["s_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=cols,
        linewidth=2,
        linestyle="dashed",
    )
    ax1.annotate(
        "S",
        ((times[slices["s_ind"]] + times[slices["sc_ind"]]) / 2, ymax * 0.8),
        color="b",
        fontsize=fs,
    )
    ax1.vlines(
        times[slices["sc_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=cols,
        linewidth=2,
        linestyle="dashed",
    )
    ax1.annotate(
        "Sc",
        ((times[slices["sc_ind"]] + times[slices["coda_ind"]]) / 2, ymax * 0.8),
        color="b",
        fontsize=fs,
    )
    ax1.vlines(
        times[slices["coda_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colc,
        linewidth=2,
        linestyle="dashed",
    )
    ax1.set_xlim([0, 180])

    ax2.plot(
        times[:-1],
        np.concatenate(
            (stc_N["R"], stc_P["R"], stc_Pc["R"], stc_S["R"], stc_Sc["R"], stc_C["R"])
        ),
        "k",
    )
    if minmax:
        ax2.set_ylim([min_val, max_val])
    ymin, ymax = ax2.get_ylim()
    ax2.vlines(
        times[slices["p_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colp,
        linewidth=2,
        linestyle="dashed",
    )
    ax2.annotate(
        "P",
        ((times[slices["p_ind"]] + times[slices["pc_ind"]]) / 2, ymax * 0.8),
        color="r",
        fontsize=fs,
    )
    ax2.vlines(
        times[slices["pc_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colp,
        linewidth=2,
        linestyle="dashed",
    )
    ax2.annotate(
        "Pc",
        ((times[slices["pc_ind"]] + times[slices["s_ind"]]) / 2, ymax * 0.8),
        color="r",
        fontsize=fs,
    )
    ax2.vlines(
        times[slices["s_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=cols,
        linewidth=2,
        linestyle="dashed",
    )
    ax2.annotate(
        "S",
        ((times[slices["s_ind"]] + times[slices["sc_ind"]]) / 2, ymax * 0.8),
        color="b",
        fontsize=fs,
    )
    ax2.vlines(
        times[slices["sc_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=cols,
        linewidth=2,
        linestyle="dashed",
    )
    ax2.annotate(
        "Sc",
        ((times[slices["sc_ind"]] + times[slices["coda_ind"]]) / 2, ymax * 0.8),
        color="b",
        fontsize=fs,
    )
    ax2.vlines(
        times[slices["coda_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colc,
        linewidth=2,
        linestyle="dashed",
    )

    ax3.plot(
        times[:-1],
        np.concatenate(
            (stc_N["T"], stc_P["T"], stc_Pc["T"], stc_S["T"], stc_Sc["T"], stc_C["T"])
        ),
        "k",
    )
    if minmax:
        ax3.set_ylim([min_val, max_val])
    ymin, ymax = ax3.get_ylim()
    ax3.vlines(
        times[slices["p_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colp,
        linewidth=2,
        linestyle="dashed",
    )
    ax3.annotate(
        "P",
        ((times[slices["p_ind"]] + times[slices["pc_ind"]]) / 2, ymax * 0.8),
        color="r",
        fontsize=fs,
    )
    ax3.vlines(
        times[slices["pc_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colp,
        linewidth=2,
        linestyle="dashed",
    )
    ax3.annotate(
        "Pc",
        ((times[slices["pc_ind"]] + times[slices["s_ind"]]) / 2, ymax * 0.8),
        color="r",
        fontsize=fs,
    )
    ax3.vlines(
        times[slices["s_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=cols,
        linewidth=2,
        linestyle="dashed",
    )
    ax3.annotate(
        "S",
        ((times[slices["s_ind"]] + times[slices["sc_ind"]]) / 2, ymax * 0.8),
        color="b",
        fontsize=fs,
    )
    ax3.vlines(
        times[slices["sc_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=cols,
        linewidth=2,
        linestyle="dashed",
    )
    ax3.annotate(
        "Sc",
        ((times[slices["sc_ind"]] + times[slices["coda_ind"]]) / 2, ymax * 0.8),
        color="b",
        fontsize=fs,
    )
    ax3.vlines(
        times[slices["coda_ind"]],
        ymin * pick_height,
        ymax * pick_height,
        color=colc,
        linewidth=2,
        linestyle="dashed",
    )

    ax3.set_xlabel("Time after %s [s]" % startT, fontsize=fs)
    ax1.set_ylabel("Velocity [nm/s] \n\n Z", fontsize=fs)
    ax2.set_ylabel("Velocity [nm/s] \n\n R", fontsize=fs)
    ax3.set_ylabel("Velocity [nm/s] \n\n T", fontsize=fs)
    plt.show()


def compute_Parrival(ev_OT, dist, ev_dep):
    p, s, warn = snsn_tt.SNSNtt(velmod="snsn", depth=ev_dep, km=dist)
    if warn:
        print(warn)
    tt_p = p[0][0]
    arrT_p = ev_OT + datetime.timedelta(seconds=tt_p)
    return arrT_p


def compute_Sarrival(ev_OT, dist, ev_dep):
    # if network in ["NS", "NO"]:
    if False:
        depths = [0, 12, 12, 23, 23, 31, 31, 50, 50, 80, 80, 6371]
        p_vels = [6.2, 6.2, 6.6, 6.6, 7.1, 7.1, 8.05, 8.05, 8.25, 8.25, 8.5, 12.75]
        s_vels = [
            3.58,
            3.58,
            3.81,
            3.81,
            4.10,
            4.10,
            4.65,
            4.65,
            4.77,
            4.77,
            4.91,
            7.37,
        ]
        vmod = [depths, p_vels, s_vels]
        try:
            p, s, warn = snsn_tt.SNSNtt(velmod=vmod, depth=ev_dep, km=dist, iter=60)
        except Exception as e:
            # Something went wrong print error message and exit
            print("Caught an  exception: %s" % (str(e)))
        else:
            if warn:
                print(warn)

        tt_s = s[0][0]
        arrT_s = ev_OT + datetime.timedelta(seconds=tt_s)
        return arrT_s

    else:
        p, s, warn = snsn_tt.SNSNtt(velmod="snsn", depth=ev_dep, km=dist)
        if warn:
            print(warn)
        tt_s = s[0][0]
        arrT_s = ev_OT + datetime.timedelta(seconds=tt_s)
        return arrT_s


def is_dst(OT, timezone="UTC"):
    dt = datetime.datetime(OT.year, OT.month, OT.day, OT.hour, OT.minute, OT.second)
    timezone = pytz.timezone(timezone)
    try:
        timezone_aware_date = timezone.localize(dt, is_dst=None)
    except Exception as e:
        print("Unable to assert if Daylight Saving Time", e)
        if OT.month in [1, 2, 3, 11, 12]:
            return False
        else:
            return True
    return timezone_aware_date.tzinfo._dst.seconds != 0


def kiru_active_hour(OT):
    dst = is_dst(OT, timezone="Europe/Stockholm")
    hr = int(OT.hour)
    mi = int(OT.minute)
    return (dst and hr == 23 and mi >= 10 and mi <= 40) or (
        not dst and hr == 0 and mi >= 10 and mi <= 40
    )


def malm_active_hour(OT):
    dst = is_dst(OT, timezone="Europe/Stockholm")
    hr = int(OT.hour)
    mi = int(OT.minute)
    return (dst and ((hr == 22 and mi <= 30) or (hr == 21 and mi > 55))) or (
        not dst and ((hr == 23 and mi <= 30) or (hr == 22 and mi > 55))
    )


def dist_kiru(ev_lat, ev_lon):
    dist = LocationDist(
        2, [ev_lat, ev_lon], [mines["kiru"][0] * d2r, mines["kiru"][1] * d2r], -1
    )[0]
    return dist


def dist_malm(ev_lat, ev_lon):
    dist = LocationDist(
        2, [ev_lat, ev_lon], [mines["malm"][0] * d2r, mines["malm"][1] * d2r], -1
    )[0]
    return dist


def dist_aitik(ev_lat, ev_lon):
    dist = LocationDist(
        2, [ev_lat, ev_lon], [mines["aiti"][0] * d2r, mines["aiti"][1] * d2r], -1
    )[0]
    return dist


def in_kiru(ev_lat, ev_lon, sil=False):
    dist = LocationDist(
        2, [ev_lat, ev_lon], [mines["kiru"][0] * d2r, mines["kiru"][1] * d2r], -1
    )[0]

    # Insert condition for the mines in Mertainen and Svappavaara.
    dist_mert = LocationDist(
        2, [ev_lat, ev_lon], [mines["mert"][0] * d2r, mines["mert"][1] * d2r], -1
    )[0]

    dl = 6
    if sil:
        if ev_lon / d2r > 20:
            return dist < dist_lim_mines_sil and dist_mert > dl
        else:
            return dist < 15
    else:
        if ev_lon / d2r > 20:
            return dist < dist_lim_mines_com and dist_mert > dl
        else:
            return dist < 12.5


def in_malm(ev_lat, ev_lon, sil=False):
    dist = LocationDist(
        2, [ev_lat, ev_lon], [mines["malm"][0] * d2r, mines["malm"][1] * d2r], -1
    )[0]

    # Insert condition for the mine in Aitik, Gällivare.
    dist_aiti = LocationDist(
        2, [ev_lat, ev_lon], [mines["aiti"][0] * d2r, mines["aiti"][1] * d2r], -1
    )[0]

    dl = 6
    if sil:
        return dist < dist_lim_mines_sil and dist_aiti > dl
    else:
        return dist < dist_lim_mines_com and dist_aiti > dl


def in_ren_kan_bjo(ev_lat, ev_lon):
    dist_lim = 3
    dist_rens = LocationDist(
        2, [ev_lat, ev_lon], [mines["rens"][0] * d2r, mines["rens"][1] * d2r], -1
    )[0]
    dist_kank = LocationDist(
        2, [ev_lat, ev_lon], [mines["kank"][0] * d2r, mines["kank"][1] * d2r], -1
    )[0]
    dist_bjor = LocationDist(
        2, [ev_lat, ev_lon], [mines["bjor"][0] * d2r, mines["bjor"][1] * d2r], -1
    )[0]

    return np.any(np.array([dist_rens, dist_kank, dist_bjor]) < dist_lim)


def stas_win_radius(ev_coord, phases):
    sta_d = sta_dict()
    for sta in stas:
        # Exclude stations with four classes and generally noisy stations.
        if sta in stas_4cl + stas_decom + ["up1", "for", "fab", "gno"]:
            continue
        sta_coord = [float(sta_d[sta][0]) * d2r, float(sta_d[sta][1]) * d2r]
        dist = LocationDist(2, ev_coord, sta_coord, -1)[0]
        if dist < 100 and sta not in [x[:-1] for x in phases]:
            i = 0
            append = True
            for sta_phase in [x[:-1] for x in phases]:
                if sta_phase in stas_old:
                    continue
                sta_phase_coord = [
                    float(sta_d[sta_phase][0]) * d2r,
                    float(sta_d[sta_phase][1]) * d2r,
                ]
                dist_phase = LocationDist(2, ev_coord, sta_phase_coord, -1)[0]
                if dist < dist_phase:
                    phases.insert(i, sta + "x")
                    append = False
                    break
                i += 1
            if append:
                phases.append(sta + "x")
    return phases


def compute_probabilities(score_ex, score_qu, score_ql, score_me, weights):
    # print(weights)
    probs = np.zeros(4)
    k = 0
    for score in [score_ex, score_qu, score_ql, score_me]:
        probs[k] = np.average(score, weights=weights)
        k += 1
    # print(np.sum(probs))
    return probs


def compute_probabilities_eventornot(score_se, score_ev, weights):
    probs = np.zeros(2)
    k = 0
    for score in [score_se, score_ev]:
        probs[k] = np.average(score, weights=weights)
        k += 1
    return probs


def month_converter(conv, fwd=True, bck=False):
    mo_fwd = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }

    mo_bck = {
        "01": "jan",
        "02": "feb",
        "03": "mar",
        "04": "apr",
        "05": "may",
        "06": "jun",
        "07": "jul",
        "08": "aug",
        "09": "sep",
        "10": "oct",
        "11": "nov",
        "12": "dec",
    }

    if fwd:
        return mo_fwd[conv]
    elif bck:
        return mo_bck[conv]


def str2datetime(date):
    if len(date) != 8:
        raise ValueError("Date string has wrong length. Format should be <yyyymmdd>")
    try:
        year = int(date[0:4])
        month = int(date[4:6])
        day = int(date[6:8])
    except Exception as e:
        print("Something wrong in date string. Format should be <yyyymmdd>", e)

    return datetime.date(year, month, day)


def time_of_day(date, time):
    if len(time) != 4 and len(time) != 6:
        raise ValueError("Time string has wrong length. Format should be <HHMM(SS)>")
    hr = int(time[0:2])
    mi = int(time[2:4])
    if len(time) == 6:
        se = int(time[4:6])
    else:
        se = 0
    return datetime.datetime(date.year, date.month, date.day, hr, mi, se)


def find_event_closest_in_time(ev_file, time, cbull=False):
    smallest_diff = 1e10
    count = 0
    if cbull:
        if not ev_file:
            return

        for ev in ev_file:
            ev_time = ev[0]
            diff = np.abs((ev_time - time).total_seconds())
            if diff < smallest_diff:
                smallest_diff = diff
                smallest_diff_line = count
            count += 1

    else:
        # Find the ID of the event closest to the given time
        count = 1
        automatic = "qual" in ev_file[0].keys()
        for ev in ev_file:
            ev_time = ev["OT"]
            diff = np.abs((ev_time - time).total_seconds())
            if diff < smallest_diff:
                smallest_diff = diff
                if automatic:
                    smallest_diff_line = count
                else:
                    smallest_diff_line = ev["id"]
            count += 1

        # If we have several automatic detections for the same event, use the one with the highest quality factor
        if automatic:
            ev_id_ref = ev_file[smallest_diff_line - 1]["file"]
            ev_ot_ref = ev_file[smallest_diff_line - 1]["OT"]
            max_qual = 0
            count = 1
            for ev in ev_file:
                ev_id = ev["file"]
                ev_ot = ev["OT"]
                if (ev_id == ev_id_ref or ev_ot == ev_ot_ref) and ev["qual"] > max_qual:
                    max_qual = ev["qual"]
                    smallest_diff_line = count
                count += 1
    return smallest_diff_line


def find_event_from_id(ev_file, Id):
    automatic = "qual" in ev_file[0].keys()
    count = 1
    line = -1
    if automatic:
        max_qual = -1
        for ev in ev_file:
            ev_id = ev["file"]
            qual = ev["qual"]
            if ev_id == Id and qual > max_qual:
                line = count
                max_qual = qual
            count += 1
    else:
        for ev in ev_file:
            ev_id = ev["file"]
            if ev_id == Id:
                line = count
                break
            count += 1
    return line


def get_evlib_path(BASE_PATH, date):
    return (
        BASE_PATH
        + "/"
        + str(date.year)
        + "/"
        + str.lower(date.strftime("%b"))
        + "/"
        + str(date.strftime("%d"))
        + "/"
        + "events.lib"
    )


def get_evaut_path(BASE_PATH, date):
    return (
        BASE_PATH
        + "/"
        + str(date.year)
        + "/"
        + str.lower(date.strftime("%b"))
        + "/"
        + str(date.strftime("%d"))
        + "/"
        + "events.aut"
    )


def sta_dict():
    with open(sta_path, "r") as fp:
        lines = fp.readlines()
    d = {}
    for line in lines:
        lin = line.split()
        try:
            sta = lin[0]
            lat = lin[1]
            lon = lin[2]
            dep = lin[3]
            site = lin[4]
        except Exception as e:
            print("Unable to fetch station information from station file", e)
            continue
        d[sta] = [lat, lon, dep, site]
    return d


def sort_uniq(List):
    new_list = []
    seen = []
    for i in range(len(List)):
        if List[i] in stas_old:
            continue
        if List[i] not in seen:
            seen.append(List[i])
        elif List[i] in seen:
            new_list.append(List[i])
    return new_list


def evclass_write_grx_sta(sta, start_date, end_date, max_dist, Kiru, Malm, grx_file):
    # Convert the start- and end-dates to datetime.date objects
    start_date = datetime.date(
        int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8])
    )
    end_date = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    curr_date = end_date

    # Maximum number of events from each class (source type)
    lim = 3000

    # Count the number of instances for each event type
    if Kiru or Malm:
        # ["ex", "ql", "me"]
        count_typs = [0, 0, 0]
    else:
        # ["ex", "qu"]
        count_typs = [0, 0]

    while curr_date >= start_date:
        # Read events.lib for curr_date
        lib_path = get_evlib_path(NAS_PATH, curr_date)
        try:
            ev_lib = read_evlib(lib_path)
        except Exception:
            # print("Unable to read events lib: %s" %lib_path, e)
            curr_date = curr_date - datetime.timedelta(days=1)
            continue

        # Loop through the events in events.lib
        for ev in ev_lib:
            ev_id = ev["file"]

            # If the event type is not one of the ones supported by the classification, continue.
            typs_poss = np.array(["ex", "qu"])
            typ = ev["type"]
            if typ not in typs_poss or (
                count_typs[np.where(typs_poss == typ)[0][0]] >= lim
            ):
                continue

            # Read the eve file for the current event
            eve_path = NAS_PATH + ev_id + ".eve"
            try:
                eve = read_eve(eve_path)
            except Exception:
                # print("Unable to read eve file: %s" %eve_path, e)
                continue

            # Proceed if the event has a picked phase at the station
            phases = list(eve["ph"].keys())
            if sta + "p" in phases or sta + "s" in phases:
                # Check that the event-station distance is within the limits set
                try:
                    dist = eve["ph"][sta + "p"]["dist"]
                except Exception:
                    dist = eve["ph"][sta + "s"]["dist"]

                if dist < max_dist:
                    OT = ev["OT"]
                    hr = int(OT.hour)
                    ev_lat = ev["lat"]
                    ev_lon = ev["lon"]

                    # Does the event originate from the Kiruna- or Malmberget mines?
                    kiru = in_kiru(ev_lat * d2r, ev_lon * d2r)
                    malm = in_malm(ev_lat * d2r, ev_lon * d2r)

                    # Case 1 - Training a model with events not associated with either of the mines in Kiruna or Malmberget.
                    if not (Kiru or Malm):
                        """
                        If:
                            * The event is associated with either Kiruna or Malmberget
                            * The event type is not blast or quake
                            * The count for the event type in question has exceeded the given limit
                            ==> continue
                        """
                        if kiru or malm:
                            continue

                        # Otherwise, write a grx line for the event
                        else:
                            lGrx = lib2grx([ev])
                            write_grx(2, lGrx)
                            if typ == "ex":
                                count_typs[0] += 1
                            elif typ == "qu":
                                count_typs[1] += 1

                    # Case 2 - Training a model with events associated with the mine in Kiruna.
                    elif Kiru:
                        typs_poss = np.array(["ex", "ql", "me"])
                        """
                        If:
                            * The event is not associated with the Kiruna mine
                            * The event type is not a blast or mining-induced
                            * The event type is mining-induced but the origin time is during the typical blasting hour for Kiruna
                            * The count for the event type in question has exceeded the given limit
                            ==> continue
                        """
                        if (
                            (not kiru)
                            or (typ not in typs_poss)
                            or (typ != "ex" and hr in [0, 23])
                            or (count_typs[np.where(typs_poss == typ)[0][0]] >= lim)
                        ):
                            continue

                        # Otherwise, write a grx line for the event
                        else:
                            lGrx = lib2grx([ev])
                            write_grx(2, lGrx)
                            if typ == "ex":
                                count_typs[0] += 1
                            elif typ == "ql":
                                count_typs[1] += 1
                            elif typ == "me":
                                count_typs[2] += 1

                    # Case 3 - Training a model with events associated with the mine in Malmberget.
                    elif Malm:
                        typs_poss = np.array(["ex", "ql", "me"])
                        """
                        If:
                            * The event is not associated with the Malmberget mine
                            * The event type is not a blast or mining-induced
                            * The event type is mining-induced but the origin time is during the typical blasting hour for Malmberget
                            * The count for the event type in question has exceeded the given limit
                            ==> continue
                        """
                        if (
                            (not malm)
                            or (typ not in typs_poss)
                            or (typ != "ex" and hr in [22, 23])
                            or (count_typs[np.where(typs_poss == typ)[0][0]] >= lim)
                        ):
                            continue

                        # Otherwise, write a grx line for the event
                        else:
                            lGrx = lib2grx([ev])
                            write_grx(2, lGrx)
                            if typ == "ex":
                                count_typs[0] += 1
                            elif typ == "ql":
                                count_typs[1] += 1
                            elif typ == "me":
                                count_typs[2] += 1

                # If the distance between the station and the event is too big, continue.
                else:
                    continue

            # If the event doesn"t have a picked phase at the station, continue.
            else:
                continue

        curr_date = curr_date - datetime.timedelta(days=1)


def evclass_data_pprocess(sta, grx_file, wf_file):
    sta_p = sta + "p"
    sta_s = sta + "s"

    # Read the input grx file
    with open(grx_file, "r") as fp:
        grxlines = fp.readlines()

    grx = ReadGrx(grxlines)

    n_evs = len(grxlines)
    n_cols = 3 * len(f_lo) * n_tws + 1
    data = np.empty(shape=(0, n_cols))
    tmp = np.empty(shape=(1, n_cols))
    for i in range(0, n_evs):
        print("%s -- %d/%d" % (sta, i + 1, n_evs))

        typ_str = grxlines[i][-3:-1]
        try:
            typ = type_conv[typ_str]
        except Exception as e:
            print("Unable to convert type string to integer", e)
            typ = type_conv[grxlines[i][-2:]]

        eve_path = NAS_PATH + grx["ev"][i]["ID"] + ".eve"
        try:
            eve = read_eve(eve_path)
        except Exception as e:
            print("Unable to read eve file: %s" % eve_path, e)
            continue

        OT = eve["OT"]
        dep = eve["dep"]
        phases = list(eve["ph"].keys())
        ev_coord = [eve["lat"] * d2r, eve["lon"] * d2r]
        sta_d = sta_dict()
        sta_coord = [float(sta_d[sta][0]) * d2r, float(sta_d[sta][1]) * d2r]
        dist = LocationDist(2, ev_coord, sta_coord, -1)[0]
        staU = smd_client.staMap(sta, 1)

        if sta_p in phases and sta_s in phases:
            arrT_p = UTCDateTime(eve["ph"][sta_p]["arrT"])
            arrT_s = UTCDateTime(eve["ph"][sta_s]["arrT"])
        elif sta_p in phases and sta_s not in phases:
            arrT_p = UTCDateTime(eve["ph"][sta_p]["arrT"])
            arrT_s = UTCDateTime(compute_Sarrival(OT, dist, dep))
        elif sta_s in phases and sta_p not in phases:
            arrT_s = UTCDateTime(eve["ph"][sta_s]["arrT"])
            arrT_p = UTCDateTime(compute_Parrival(OT, dist, dep))
        else:
            continue

        try:
            tmp[0, :-1] = generate_waveformdata(
                arrT_p=arrT_p,
                arrT_s=arrT_s,
                staU=staU,
                network="UP",
                ev_lat=ev_coord[0],
                ev_lon=ev_coord[1],
            )[1]
            tmp[0, -1] = int(typ)
            data = np.append(data, tmp, axis=0)
        except Exception as e:
            print("Problem generating waveform data", e)
            continue

        np.save(wf_file, data)
        # np.save(save_dir + "/%s" %(infile.split("/")[-1][:-4]), data)
    return data


def evclass_train_model(sta, data, Kiru, Malm, save_dir):
    from tensorflow import keras

    # Case 1 - Training a model with events not associated with either of the mines in Kiruna or Malmberget.
    if not (Kiru or Malm):
        # Delete any events with an unsupported label from the dataset
        ind_del = []
        for i in range(data.shape[0]):
            if data[i][-1] not in [0, 1]:
                ind_del.append(i)
        data = np.delete(data, ind_del, axis=0)

        # Create a pandas dataframe containing the data
        df = pd.DataFrame(data, columns=cols)
        df = df.dropna()

        # Split the data into features and labels
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Save the features to file (to be able to normalize new data)
        X.to_pickle(save_dir + "/%s_feats.pkl" % sta)

        # Print the number of examples with each label
        print("Number of different types in the whole dataset before over-sampling:")
        print(y.value_counts())
        print("--------------------------------------------------------")

        # Normalize the features
        X = (X - X.mean()) / X.std()

        # If needed, oversample the under-represented class
        ovsamp = {}

        if y.value_counts()[0] / y.value_counts()[1] < 0.8:
            ovsamp[0] = int(np.round(0.8 * y.value_counts()[1]))
        elif y.value_counts()[1] / y.value_counts()[0] < 0.8:
            ovsamp[1] = int(np.round(0.8 * y.value_counts()[0]))
        try:
            oversample = SMOTE(sampling_strategy=ovsamp)
            X, y = oversample.fit_resample(X, y)
        except Exception as e:
            print("No oversampling needed", e)

        print("Number of different types in the whole dataset after over-sampling:")
        print(y.value_counts())

        # Split into training and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        y_train = y_train.astype("int32")
        y_test = y_test.astype("int32")
        print("Number of different types in y_test:")
        print(y_test.value_counts())
        print("--------------------------------------------------------")

        # Construct the classification model
        model = keras.models.Sequential(
            [
                keras.layers.Dense(
                    256, activation="relu", input_shape=(X_train.shape[-1],)
                ),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        print(model.summary())

        # Define evaluation metrics
        metrics = [
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="binary_crossentropy",
            metrics=metrics,
        )

        # Define callbacks
        callback = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )

        # Train the model
        results = model.fit(
            X_train,
            y_train,
            batch_size=64,
            epochs=500,
            verbose=1,
            validation_split=0.25,
            callbacks=[callback],
        )

        # Plot some results
        epoch = range(len(results.history["val_recall"]))
        train_loss = results.history["loss"]
        val_loss = results.history["val_loss"]

        fig, ax = plt.subplots()
        ax.set_title("Loss")
        ax.plot(epoch, train_loss, color="b", label="train")
        ax.plot(epoch, val_loss, color="r", label="validation")
        ax.legend()
        fig.savefig(save_dir + "/%s_training.png" % sta)

        # Print confusion matrix
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print(confusion_matrix(y_test, y_pred))
        print("------------------------------------------------\n")

        # Print and save classification report
        print(classification_report(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)
        with open(save_dir + "/%s_report.pkl" % sta, "wb") as handle:
            pickle.dump(report, handle)

        # Print Cohen-Kappa score
        print("------------------------------------------------\n")
        cohen_score = cohen_kappa_score(y_test, y_pred)
        print("CKS: ", cohen_score)

        # Save the model
        model.save(save_dir + "/%s_model.h5" % sta)

    # Case 2 - Training a model with events associated with the mines in Kiruna/Malmberget.
    else:
        n_classes = 3

        # Change the labels since we only define 3 classes in the mines
        for i in range(data.shape[0]):
            if data[i][-1] == 2:
                data[i][-1] = 1
            elif data[i][-1] == 3:
                data[i][-1] = 2

        # Create a pandas dataframe containing the data
        df = pd.DataFrame(data, columns=cols)
        df = df.dropna()

        # Split the data into features and labels
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        if Kiru:
            X.to_pickle(save_dir + "/%s_kiru_feats.pkl" % sta)
        elif Malm:
            X.to_pickle(save_dir + "/%s_malm_feats.pkl" % sta)

        # Print the number of examples with each label
        print("Number of different types in the whole dataset before over-sampling:")
        print(y.value_counts())
        print("--------------------------------------------------------")

        # Normalize the features
        X = (X - X.mean()) / X.std()

        # If needed, oversample the under-represented class
        ovsamp = {}

        if y.value_counts()[0] / y.value_counts()[1] < 0.8:
            ovsamp[0] = int(np.round(0.8 * y.value_counts()[1]))
        elif y.value_counts()[1] / y.value_counts()[0] < 0.8:
            ovsamp[1] = int(np.round(0.8 * y.value_counts()[0]))
        elif y.value_counts()[2] / y.value_counts()[0] < 0.8:
            ovsamp[2] = int(np.round(0.8 * y.value_counts()[0]))
        try:
            oversample = SMOTE(sampling_strategy=ovsamp)
            X, y = oversample.fit_resample(X, y)
        except Exception as e:
            print("No oversampling needed", e)

        print("Number of different types in the whole dataset after over-sampling:")
        print(y.value_counts())

        # Split into training and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Convert labels ("0"-"2") to one-hot encodings, "0" = (1, 0, ... 0) and so on
        y_train_onehot = keras.utils.to_categorical(y_train, n_classes)
        y_test_onehot = keras.utils.to_categorical(y_test, n_classes)

        print("Number of different types in y_test:")
        print(y_test.value_counts())
        print("--------------------------------------------------------")

        # Construct the classification model
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    256, activation="relu", input_shape=(X_train.shape[-1],)
                ),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )

        # Define evaluation metrics
        metrics = [
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        # Define callbacks
        callback = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )

        # Train the model
        results = model.fit(
            X_train,
            y_train_onehot,
            batch_size=64,
            epochs=500,
            verbose=1,
            validation_split=0.25,
            callbacks=[callback],
        )

        # Plot some results
        epoch = range(len(results.history["val_recall"]))
        train_loss = results.history["loss"]
        val_loss = results.history["val_loss"]

        fig, ax = plt.subplots()
        ax.set_title("Loss")
        ax.plot(epoch, train_loss, color="b", label="train")
        ax.plot(epoch, val_loss, color="r", label="validation")
        ax.legend()
        if Kiru:
            fig.savefig(save_dir + "/%s_kiru_training.png" % sta)
        elif Malm:
            fig.savefig(save_dir + "/%s_malm_training.png" % sta)

        # calculate predictions for test set
        y_pred = model.predict(X_test, batch_size=32)

        # convert back to class labels (0-9)
        y_test_cl = np.argmax(y_test_onehot, axis=1)
        y_pred_cl = np.argmax(y_pred, axis=1)

        # Print confusion matrix
        print(confusion_matrix(y_test_cl, y_pred_cl))
        print("------------------------------------------------\n")

        # Print and save classification report
        print(classification_report(y_test_cl, y_pred_cl))
        report = classification_report(y_test_cl, y_pred_cl, output_dict=True)
        if Kiru:
            with open(save_dir + "/%s_kiru_report.pkl" % sta, "wb") as handle:
                pickle.dump(report, handle)
        elif Malm:
            with open(save_dir + "/%s_malm_report.pkl" % sta, "wb") as handle:
                pickle.dump(report, handle)

        # Print Cohen-Kappa score
        print("------------------------------------------------\n")
        cohen_score = cohen_kappa_score(y_test_cl, y_pred_cl)
        print("CKS: ", cohen_score)

        # Save the model
        if Kiru:
            model.save(save_dir + "/%s_kiru_model.h5" % sta)
        elif Malm:
            model.save(save_dir + "/%s_malm_model.h5" % sta)


def classification_map(ev_lat, ev_lon, seen, dists, sta_pred, pred_typ, conf, distance):
    import pygmt

    fig = pygmt.Figure()

    rang_lat = 2.5
    rang_lon = 5.5
    fig.coast(
        shorelines=False,
        region=[
            ev_lon - rang_lon,
            ev_lon + rang_lon,
            ev_lat - rang_lat,
            ev_lat + rang_lat,
        ],
        projection="M20c",
        water="lightskyblue",
        land="oldlace",
        borders="1/1p",
        frame="a",
    )

    # Plot the stations
    sta_file = sta_path
    with open(sta_file) as fp:
        stalist = fp.readlines()

    # Stations with predictions
    sta_typ = []
    sta_conf = []
    sta_lat = []
    sta_lon = []
    sta_cod = []
    sta_alt = []
    # Stations with no phase picked
    sta_lat_n = []
    sta_lon_n = []
    sta_cod_n = []
    sta_alt_n = []
    # Stations with a phase picked but no prediction
    sta_typ_e = []
    sta_lat_e = []
    sta_lon_e = []
    sta_cod_e = []
    sta_alt_e = []
    # Stations with a phase picked but too far awya
    sta_cod_d = []
    sta_lat_d = []
    sta_lon_d = []
    for sta in stalist:
        st = sta.split()
        cod = st[0]
        if cod in stas_decom:
            continue
        STA = smd_client.staMap(cod, 1)
        if STA is None or cod in ["ASK"]:
            STA = cod
        if STA not in list(sta_pred.keys()):
            sta_cod_n.append(STA)
            sta_lat_n.append(float(st[1]))
            sta_lon_n.append(float(st[2]))
            sta_alt_n.append(float(st[3]))
        else:
            if sta_pred[STA][0] not in ["ex", "qu", "ql", "me"]:
                if sta_pred[STA][0][0] == "E":
                    sta_cod_e.append(STA)
                    sta_lat_e.append(float(st[1]))
                    sta_lon_e.append(float(st[2]))
                    sta_alt_e.append(float(st[3]))
                    sta_typ_e.append(sta_pred[STA][0])
                else:
                    sta_cod_d.append(STA)
                    sta_lat_d.append(float(st[1]))
                    sta_lon_d.append(float(st[2]))
            else:
                sta_cod.append(STA)
                sta_lat.append(float(st[1]))
                sta_lon.append(float(st[2]))
                sta_alt.append(float(st[3]))
                sta_typ.append(sta_pred[STA][0])
                sta_conf.append(int(sta_pred[STA][1]))

    s = {
        "code": sta_cod,
        "latitude": sta_lat,
        "longitude": sta_lon,
        "altitude": sta_alt,
        "type": sta_typ,
        "confidence": sta_conf,
    }
    station = pd.DataFrame(data=s)

    s_n = {
        "code": sta_cod_n,
        "latitude": sta_lat_n,
        "longitude": sta_lon_n,
        "altitude": sta_alt_n,
    }
    station_n = pd.DataFrame(data=s_n)

    s_e = {
        "code": sta_cod_e,
        "latitude": sta_lat_e,
        "longitude": sta_lon_e,
        "altitude": sta_alt_e,
        "type": sta_typ_e,
    }
    station_e = pd.DataFrame(data=s_e)

    s_d = {"code": sta_cod_d, "latitude": sta_lat_d, "longitude": sta_lon_d}
    station_d = pd.DataFrame(data=s_d)

    fig.plot(
        x=station.longitude,
        y=station.latitude,
        style="t0.5c",
        color="darkblue",
        pen="black",
        label="Pick Available".replace(" ", r"\040"),
    )

    fig.plot(
        x=station_n.longitude,
        y=station_n.latitude,
        style="t0.5c",
        color="darkgrey",
        pen="black",
        label="No Pick Available".replace(" ", r"\040"),
    )

    fig.text(
        text=s["code"],
        x=s["longitude"],
        y=[y - 0.07 - (68 - y) * 0.001 for y in s["latitude"]],
        font="6p,Helvetica-Bold,black",
    )

    fig.text(
        text=s["type"],
        x=[x - 0.08 for x in s["longitude"]],
        y=[y - 0.12 - (68 - y) * 0.002 for y in s["latitude"]],
        font="6p,Helvetica-Bold,darkgreen",
    )

    fig.text(
        text=s["confidence"],
        x=[x + 0.08 for x in s["longitude"]],
        y=[y - 0.12 - (68 - y) * 0.002 for y in s["latitude"]],
        font="5p,Helvetica-Bold,darkgreen",
    )

    fig.text(
        text=s_n["code"],
        x=s_n["longitude"],
        y=[y - 0.07 - (68 - y) * 0.002 for y in s_n["latitude"]],
        font="6p,Helvetica-Bold,black",
    )

    if len(station_e) > 0:
        fig.plot(
            x=station_e.longitude,
            y=station_e.latitude,
            style="t0.5c",
            color="red",
            pen="black",
            label="Not Used".replace(" ", r"\040"),
        )

        fig.text(
            text=s_e["code"],
            x=s_e["longitude"],
            y=[y - 0.07 - (68 - y) * 0.002 for y in s_e["latitude"]],
            font="6p,Helvetica-Bold,black",
        )

        fig.text(
            text=s_e["type"],
            x=[x for x in s_e["longitude"]],
            y=[y - 0.13 - (68 - y) * 0.002 for y in s_e["latitude"]],
            font="6p,Helvetica-Bold,red",
        )

    if len(station_d) > 0:
        fig.plot(
            x=station_d.longitude,
            y=station_d.latitude,
            style="t0.5c",
            color="darkblue",
            pen="black",
        )

        fig.text(
            text=s_d["code"],
            x=s_d["longitude"],
            y=[y - 0.07 - (68 - y) * 0.002 for y in s_d["latitude"]],
            font="6p,Helvetica-Bold,black",
        )

    fig.legend()

    # Plot the event location and distance circles
    fig.plot(x=ev_lon, y=ev_lat, style="a0.4c", color="red")

    fig.text(
        text=pred_typ, x=ev_lon - 0.08, y=ev_lat - 0.07, font="6p,Helvetica-Bold,red"
    )

    fig.text(
        text=str(conf), x=ev_lon + 0.08, y=ev_lat - 0.07, font="5p,Helvetica-Bold,red"
    )

    if distance == []:
        dist_max = np.max([150, np.max(dists)])
    else:
        dist_max = distance[1]

    """
    fig.plot(
            data=[[ev_lon, ev_lat, dist_min*2]], 
            style="E-", 
            pen="1.5p,red"
    )
    """

    fig.plot(
        data=[[ev_lon, ev_lat, dist_max * 2]],
        style="E-",
        pen="1.5p,darkgreen",
        transparency=30,
    )

    fig.show()
    fig.savefig(evclass_path + "/classification_map.png")


def in_combull(ev_id):
    OT_thres = 10
    dist_thres = 60

    evb_path = BASE_PATH + ev_id + ".evb"
    evb = read_eve(evb_path)
    evaut_OT = evb["OT"]
    evaut_lat = evb["lat"]
    evaut_lon = evb["lon"]

    out = ["ot", "lat", "lon", "dep", "mag", "scid", "ewid", "region", "mask"]
    start = datetime.datetime(evaut_OT.year, evaut_OT.month, evaut_OT.day, 0, 0, 0)
    end = datetime.datetime(evaut_OT.year, evaut_OT.month, evaut_OT.day, 23, 59, 59)
    evs = combull.catalog(ot=[start, end], out=out)

    in_combull = False
    for ev in evs:
        evcbl_OT = ev[0]
        evcbl_lat = ev[1]
        evcbl_lon = ev[2]
        evcbl_mask = ev[-1]

        # Epicentral distance
        dist_cbl = LocationDist(
            2,
            [evaut_lat * d2r, evaut_lon * d2r],
            [evcbl_lat * d2r, evcbl_lon * d2r],
            -1,
        )[0]

        # Origin time difference
        diff_OT_cbl = np.abs((evaut_OT - evcbl_OT).total_seconds())

        if diff_OT_cbl <= OT_thres and dist_cbl <= dist_thres and evcbl_mask > 0:
            in_combull = True
            break

    return in_combull


def in_silbull(ev_id):
    OT_thres = 10
    dist_thres = 60

    evb_path = BASE_PATH + ev_id + ".evb"
    evb = read_eve(evb_path)
    evaut_OT = evb["OT"]
    evaut_lat = evb["lat"]
    evaut_lon = evb["lon"]

    out = ["ot", "lat", "lon", "dep", "mag", "typ", "mask"]
    start = datetime.datetime(
        evaut_OT.year, evaut_OT.month, evaut_OT.day, 0, 0, 0
    ) - datetime.timedelta(days=1)
    end = datetime.datetime(
        evaut_OT.year, evaut_OT.month, evaut_OT.day, 23, 59, 59
    ) + datetime.timedelta(days=1)

    host = "seppl"
    user = "bulletin"
    passwd = ""
    db = "dev"
    table = "ann_evaut"

    db_conn = sil.connect(host=host, user=user, passwd=passwd, db=db, table=table)

    evs = sil.catalog(db_conn=db_conn, ot=[start, end], mask=1, out=out)

    in_silbull = False
    for ev in evs:
        evsil_OT = ev[0]
        evsil_lat = ev[1]
        evsil_lon = ev[2]
        evsil_mask = ev[-1]

        # Epicentral distance
        dist_sil = LocationDist(
            2,
            [evaut_lat * d2r, evaut_lon * d2r],
            [evsil_lat * d2r, evsil_lon * d2r],
            -1,
        )[0]

        # Origin time difference
        diff_OT_sil = np.abs((evaut_OT - evsil_OT).total_seconds())

        if diff_OT_sil <= OT_thres and dist_sil <= dist_thres and evsil_mask == 1:
            in_silbull = True
            break

    sil.close(db_conn)
    return in_silbull
