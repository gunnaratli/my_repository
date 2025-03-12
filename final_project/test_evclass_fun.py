#!/home/gunnar/anaconda3/envs/obspy/bin/python

import evclass_fun as ec
import silio
import numpy as np
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
import datetime
import pytest
import warnings
warnings.simplefilter("ignore")


# Test generate_waveformdata
test_evornot = np.load("/home/gunnar/python/test_generate_waveformdata_evornot.npy")
test_evclass = np.load("/home/gunnar/python/test_generate_waveformdata_evclass.npy")
arrT_p       = UTCDateTime("2025-03-11T11:03:02.060000Z")
arrT_s       = UTCDateTime("2025-03-11T11:03:09.540000Z")
staU         = "NRAU"
network      = "UP"
ev_lat       = 59.3493*ec.d2r
ev_lon       = 13.9928*ec.d2r
def test_generate_waveformdata_evornot():
	assert np.array_equal(ec.generate_waveformdata(arrT_p, arrT_s, staU, network, ev_lat, ev_lon)[0], test_evornot)
def test_generate_waveformdata_evclass():
	assert np.array_equal(ec.generate_waveformdata(arrT_p, arrT_s, staU, network, ev_lat, ev_lon)[1], test_evclass)


# Test remove_IR
test_vel = read("/home/gunnar/python/test_IR_vel.mseed")
test_raw = read("/home/gunnar/python/test_IR_raw.mseed")
def test_remove_IR_1():
    assert np.array_equal(ec.remove_IR(test_raw)[0].data, test_vel[0].data)


# Test rms
@pytest.mark.parametrize("arr, rms", [
	(np.array([1,2]), np.sqrt(10)/2),
	(np.array([15,16]), np.sqrt(962)/2)
	])
def test_rms(arr, rms):
	assert ec.rms(arr) == rms


# Test is_dst
@pytest.mark.parametrize("datetime, is_dst", [
	(datetime.datetime(2024,3,30,0,0,0), False),
	(datetime.datetime(2024,3,31,5,0,0), True)
	])
def test_is_dst(datetime, is_dst):
	assert ec.is_dst(datetime, timezone="Europe/Stockholm") == is_dst


# Test kiru_active_hour
@pytest.mark.parametrize("OT, kiru_active_hour", [
	(datetime.datetime(2024,1,1,23,0,0), False),
	(datetime.datetime(2024,1,1,0,20,0), True),
	(datetime.datetime(2024,1,1,1,0,0), False)
	])
def test_kiru_active_hour(OT, kiru_active_hour):
	assert ec.kiru_active_hour(OT) == kiru_active_hour


# Test malm_active_hour
@pytest.mark.parametrize("OT, malm_active_hour", [
	(datetime.datetime(2024,1,1,23,0,0), True),
	(datetime.datetime(2024,1,1,0,20,0), False),
	(datetime.datetime(2024,1,1,1,0,0), False)
	])
def test_malm_active_hour(OT, malm_active_hour):
	assert ec.malm_active_hour(OT) == malm_active_hour


# Test in_kiru
@pytest.mark.parametrize("ev_lat, ev_lon, sil, in_kiru", [
	(67.83*ec.d2r, 20.18*ec.d2r, True, True),
	(67.20*ec.d2r, 20.30*ec.d2r, True, False)
	])
def test_in_kiru(ev_lat, ev_lon, sil, in_kiru):
	assert ec.in_kiru(ev_lat, ev_lon, sil) == in_kiru


# Test in_malm
@pytest.mark.parametrize("ev_lat, ev_lon, sil, in_malm", [
	(67.19*ec.d2r, 20.70*ec.d2r, True, True),
	(67.90*ec.d2r, 20.24*ec.d2r, True, False)
	])
def test_in_malm(ev_lat, ev_lon, sil, in_malm):
	assert ec.in_malm(ev_lat, ev_lon, sil) == in_malm


# Test compute_probabilities
@pytest.mark.parametrize("score_ex, score_qu, score_ql, score_me, weights, probs", [
	([0.5, 0.5], [0.5, 0.5], [0, 0], [0, 0], [1, 1], [0.5, 0.5, 0, 0]),
	([0.25], [0.25], [0.25], [0.25], [1], [0.25, 0.25, 0.25, 0.25]),
	([0.25, 1], [0.50, 0], [0, 0], [0.25, 0], [1, 1], [0.625, 0.25, 0, 0.125])
	])
def test_compute_probabilities(score_ex, score_qu, score_ql, score_me, weights, probs):
	assert np.array_equal(np.array(ec.compute_probabilities(score_ex, score_qu, score_ql, score_me, weights)), np.array(probs))


# Test compute_probabilities_eventornot
@pytest.mark.parametrize("score_se, score_ev, weights, probs", [
	([0.5, 0.5], [0.5, 0.5], [1, 1], [0.5, 0.5]),
	([0.25], [0.75], [1], [0.25, 0.75])
	])
def test_compute_probabilities_eventornot(score_se, score_ev, weights, probs):
	assert np.array_equal(np.array(ec.compute_probabilities_eventornot(score_se, score_ev, weights)), np.array(probs))


# Test str2datetime
@pytest.mark.parametrize("date, str2datetime", [
	("20240301", datetime.date(2024,3,1)),
	("20211117", datetime.date(2021,11,17))
	])
def test_str2datetime(date, str2datetime):
	assert ec.str2datetime(date) == str2datetime


# Test time_of_day
@pytest.mark.parametrize("date, time, datetime", [
	(datetime.date(2024,3,1), "1105", datetime.datetime(2024,3,1,11,5,0)),
	(datetime.date(2021,11,17), "050505", datetime.datetime(2021,11,17,5,5,5))
	])
def test_time_of_day(date, time, datetime):
	assert ec.time_of_day(date, time) == datetime


# Test find_event_closest_in_time
@pytest.mark.parametrize("ev_file, time, cbull, index", [
	(silio.read_evlib("/mnt/snsn_data/eq1/2024/jan/01/events.lib"), datetime.datetime(2024,1,1,18,20,0), False, 5),
	(silio.read_evlib("/mnt/snsn_data/eq1/2024/jan/01/events.lib"), datetime.datetime(2024,1,1,7,50,0), False, 2),
	(silio.read_evlib("/mnt/snsn_data/eq1/2025/jan/01/events.lib"), datetime.datetime(2025,1,1,14,0,0), False, 8)
	])
def test_find_event_closest_in_time(ev_file, time, cbull, index):
	assert ec.find_event_closest_in_time(ev_file, time, cbull) == index


# Test find_event_from_id
@pytest.mark.parametrize("ev_file, Id, index", [
	(silio.read_evlib("/mnt/snsn_data/eq1/2024/jan/01/events.lib"), "/2024/jan/01/18:45:00/48:26:953", 5),
	(silio.read_evlib("/mnt/snsn_data/eq1/2024/jan/01/events.lib"), "/2024/jan/01/07:35:00/41:19:920", 2),
	(silio.read_evlib("/mnt/snsn_data/eq1/2025/jan/01/events.lib"), "/2025/jan/01/13:55:00/02:39:070", 8)
	])
def test_find_event_from_id(ev_file, Id, index):
	assert ec.find_event_from_id(ev_file, Id) == index


# Test get_evaut_path
@pytest.mark.parametrize("BASE_PATH, date, path", [
	("/mnt/sil/eq1", datetime.date(2024,1,1), "/mnt/sil/eq1/2024/jan/01/events.aut"),
	("/mnt/sil/eq1", datetime.date(2023,11,24), "/mnt/sil/eq1/2023/nov/24/events.aut"),
	("/mnt/sil/eq1", datetime.date(2016,2,14), "/mnt/sil/eq1/2016/feb/14/events.aut")
	])
def test_get_evaut_path(BASE_PATH, date, path):
	assert ec.get_evaut_path(BASE_PATH, date) == path


# Test get_evlib_path
@pytest.mark.parametrize("BASE_PATH, date, path", [
	("/mnt/sil/eq1", datetime.date(2024,1,1), "/mnt/sil/eq1/2024/jan/01/events.lib"),
	("/mnt/sil/eq1", datetime.date(2023,11,24), "/mnt/sil/eq1/2023/nov/24/events.lib"),
	("/mnt/sil/eq1", datetime.date(2016,2,14), "/mnt/sil/eq1/2016/feb/14/events.lib")
	])
def test_get_evlib_path(BASE_PATH, date, path):
	assert ec.get_evlib_path(BASE_PATH, date) == path

# Test sort_uniq
@pytest.mark.parametrize("lst1, lst2", [
	(["a", "aa", "aaa", "aa"], ["aa"]),
	(["a", "aa", "aaa", "aaaa"], []),
	])
def test_sort_uniq(lst1, lst2):
	assert ec.sort_uniq(lst1) == lst2


# Test month converter in forward mode
@pytest.mark.parametrize("mon_conv1, mon_conv2", [
	("jan", "01"),
	("apr", "04"),
	("dec", "12")
	])
def test_month_converter_fwd(mon_conv1, mon_conv2):
	assert ec.month_converter(mon_conv1) == mon_conv2


# Test month converter in backward mode
@pytest.mark.parametrize("mon_conv2, mon_conv1", [
	("02", "feb"),
	("05", "may"),
	("12", "dec")
	])
def test_month_converter_bck(mon_conv2, mon_conv1):
	assert ec.month_converter(mon_conv2, fwd=False, bck=True) == mon_conv1


# Test in_combull
@pytest.mark.parametrize("ev_id, in_combull", [
	("/2025/mar/12/13:05:00/12:01:979", True),
	("/2025/mar/12/12:10:00/17:49:460", False)
	])
def test_in_combull(ev_id, in_combull):
	assert ec.in_combull(ev_id) == in_combull


# # Test in_silbull
@pytest.mark.parametrize("ev_id, in_silbull", [
	("/2025/mar/12/13:05:00/12:01:979", False),
	("/2025/mar/12/12:10:00/17:49:460", False),
	("/2025/mar/10/09:10:00/13:43:960", True),
	("/2025/mar/08/05:40:00/45:29:419", False),
	("/2025/mar/08/14:30:00/35:21:859", False),
	("/2025/mar/12/11:55:00/00:37:469", True)
	])
def test_in_silbull(ev_id, in_silbull):
	assert ec.in_silbull(ev_id) == in_silbull