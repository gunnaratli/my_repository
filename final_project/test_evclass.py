#!/home/gunnar/anaconda3/envs/obspy/bin/python

from evclass import evclass
import datetime
import pytest


# Test evclass using SIL-ID for automatic SIL detections
@pytest.mark.parametrize("ev_id, classification", [
	("/2025/feb/19/15:10:00/16:44:769", [97, "qu", 97, 7]),
	])
def test_silid_auto(ev_id, classification):
	assert evclass(ev_id=ev_id,print_cl=False,search_path="/mnt/snsn_data/eq1") == classification


# Test evclass using date and time, for automatic SIL detections
@pytest.mark.parametrize("date, ev_time, classification", [
	(datetime.date(2025,2,19), "151632", [97, "qu", 97, 7]),
	])
def test_datetime_auto(date, ev_time, classification):
	assert evclass(date=date, ev_time=ev_time, print_cl=False, search_path="/mnt/snsn_data/eq1") == classification


# Test evclass using date and line number, for automatic SIL detections
@pytest.mark.parametrize("date, ev_number, classification", [
	(datetime.date(2025, 2, 19), 177, [97, "qu", 97, 7]),
	])
def test_datenumber_auto(date, ev_number, classification):
	assert evclass(date=date, ev_number=ev_number, print_cl=False, search_path="/mnt/snsn_data/eq1") == classification


# Test excluding stations
@pytest.mark.parametrize("date, ev_number, exclude, classification", [
	(datetime.date(2025, 2, 19), 177, ["STRU", "LNKU"], [96, "qu", 96, 5]),
	])
def test_datenumber_auto_excl(date, ev_number, exclude, classification):
	assert evclass(date=date, ev_number=ev_number, print_cl=False, search_path="/mnt/snsn_data/eq1", exclude=exclude) == classification


# Test including stations
@pytest.mark.parametrize("date, ev_number, include, classification", [
	(datetime.date(2025, 2, 19), 177, ["STRU", "LNKU"], [75, "qu", 75, 2]),
	])
def test_datenumber_auto_incl(date, ev_number, include, classification):
	assert evclass(date=date, ev_number=ev_number, print_cl=False, search_path="/mnt/snsn_data/eq1", include=include) == classification



# Test changing the distance range
@pytest.mark.parametrize("date, ev_number, distance, classification", [
	(datetime.date(2025, 2, 19), 177, [10, 100], [93, "qu", 93, 4]),
	])
def test_datenumber_auto_dist(date, ev_number, distance, classification):
	assert evclass(date=date, ev_number=ev_number, print_cl=False, search_path="/mnt/snsn_data/eq1", distance=distance) == classification


# Test station-specific models
@pytest.mark.parametrize("date, ev_number, sta_spec, classification", [
	(datetime.date(2025, 2, 19), 177, True, [97, "qu", 97, 7]),
	])
def test_datenumber_auto_staspec(date, ev_number, sta_spec, classification):
	assert evclass(date=date, ev_number=ev_number, print_cl=False, search_path="/mnt/snsn_data/eq1", sta_spec=sta_spec) == classification


# Test old event
@pytest.mark.parametrize("ev_id, classification", [
	("/2004/apr/04/02:00:00/05:35:339", [92, "qu", 95, 5]),
	])
def test_silid_auto_old(ev_id, classification):
	assert evclass(ev_id=ev_id,print_cl=False,search_path="/mnt/snsn_data/eq1") == classification


# Test evclass using SIL-ID for manually analyzed events
@pytest.mark.parametrize("ev_id, classification", [
	("/2025/jan/01/12:05:00/08:16:111", [97, "ql", 92, 6]),
	("/2004/may/12/23:40:00/44:45:929", [82, "qu", 93, 4]),
	("/2017/apr/04/11:55:00/58:47:128", [96, "ex", 97, 6])
	])
def test_evclass_silid_manual(ev_id, classification):
	assert evclass(ev_id=ev_id, print_cl=False, ana=True, search_path="/mnt/snsn_data/eq1") == classification