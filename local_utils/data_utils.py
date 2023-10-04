# load JSON data to GDX

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, json
import pandas as pd
import numpy as np
import logging
import datetime

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_participant_info(part_dir=None):
    """ Loads participant info from an Excel file into a directory """
    if part_dir is None:
        part_dir = 'offer_data/'
    df_participant = pd.read_excel(os.path.join(part_dir, 'participant_info.xlsx'))
    pids = list(df_participant.pid)
    participant_res = {}
    for idx, pid in enumerate(pids):
        participant_res[pid] = df_participant.ResourceList[idx].split(',')
    return participant_res

def load_resource_info(res_dir=None):
    """ Loads resource info from an Excel file into a dict of pandas dataframes """
    if res_dir is None:
        res_dir = 'offer_data/'
    resources_df = pd.read_excel(os.path.join(res_dir, 'resource_info.xlsx'),sheet_name=None)
    return resources_df

def get_time_from_system(uid, system_dir='system_data'):
    # Import system data, most relevantly timeseries (system_data['t'])
    json_file_path = os.path.join(system_dir, f'{uid}.json')
    with open(json_file_path, 'r') as f:
        system_data = json.load(f)
    return system_data['t']

def get_value_from_forecast(times, rtype, tstep=5, system_dir='system_data'):
    '''Loads the renewable and demand values from the forecast at the given timestamps.'''
    forecast = pd.read_excel(os.path.join(system_dir,'forecast.xlsx'), sheet_name=None)
    # Find the index at which the market type ends in the time range
    time_idx = [idx for idx, t in enumerate(times[0]) if t == '2'][0]
    time0 = datetime.datetime.strptime(times[0][time_idx:], '%Y%m%d%H%M')
    sheet = forecast[rtype.capitalize()]
    t_start = None
    for idx, val in enumerate(sheet['Time'].values):
        if pd.to_datetime(val) == time0:
            t_start = idx
            break
    if t_start is None:
        raise ValueError(f'Start time {time0} not found in {rtype} forecast')
    dt = int(tstep/5)
    power_mw = sheet['MW'].values[t_start:t_start+len(times)*dt:dt]
    return power_mw

def update_competitor_status(resources_df, participant_res, uid, offer_dir=None):
    """Sends a file with resource/system status to competitor directories"""
    if offer_dir is None:
        offer_dir = 'offer_data/'
    # Output the forecast for this market into the offer data directory
    times = get_time_from_system(uid)
    # TODO, the below will give incorrect forecasts in the RHF market with variable tstep
    tstep = int(times[1])-int(times[0]) 
    if tstep == 100: # Change hour step to correct minutes
        tstep = 60
    forecast_out = {}
    for rtype in ['wind', 'solar', 'demand']:
        rdict = {}
        power = get_value_from_forecast(times, rtype, tstep)
        for i, t in enumerate(times):
            rdict[t] = power[i]
        forecast_out[rtype] = rdict
    with open('./offer_data/forecast.json', "w") as f:
        json.dump(forecast_out, f, indent=4, cls=NpEncoder)
    
    # Loop through all participants and write a status file for each
    for pid, rlist in participant_res.items():
        soc_last, temp_last = 75 + 10*np.random.randn(), 30 + 3*np.random.randn()
        # Load status into a dictionary for writing to json file
        status_dict = {}
        status_dict['participant_id'] = pid
        status_dict['market_id'] = uid
        resource_dict = {'soc': soc_last, 'temp': temp_last}
        status_dict['resources'] = resource_dict
        pdir = os.path.join(offer_dir,f'participant_{pid}')
        if not os.path.exists(pdir):
            os.makedirs(pdir)
        part_file = os.path.join(pdir,'status.json')
        with open(part_file, "w") as f:
            json.dump(status_dict, f, indent=4, cls=NpEncoder)

def find_time_list(uid, mkt_config, num_fwd):
    '''Returns a list of the forward times at five-minute intervals'''
    uidx = [i for i, char in enumerate(uid) if char == '2'][0]
    t0 = datetime.datetime.strptime(uid[uidx:], '%Y%m%d%H%M')
    phys_shift = 0 # Tracker for whether there is a physical interval or not
    if 'PHYS' in mkt_config.interval_types[0]:
        t0 += datetime.timedelta(minutes=5)
        phys_shift = 1
    t_end = t0
    time_list = []
    fwd_rem = num_fwd
    idx_break = [0] # Tracks when the times interval duration changes
    for interval in mkt_config.interval_durations:
        fwd_cnt, minutes = interval[0], interval[1]
        # On the first pass through, reduce the forward count by 1 if there is a physical interval
        if len(idx_break) == 1:
            fwd_cnt -= phys_shift
        if fwd_cnt >= fwd_rem:
            t_end += datetime.timedelta(minutes=minutes*fwd_rem)
            idx_break += [idx_break[-1]+int(fwd_rem*minutes/5)]
            fwd_rem = 0
            break
        else:
            t_end += datetime.timedelta(minutes=minutes*fwd_cnt)
            idx_break += [idx_break[-1]+int(fwd_cnt*minutes/5)]
            fwd_rem -= fwd_cnt
    num_five_min_intervals = int((t_end-t0).total_seconds()/60/5)
    idx_break = idx_break[1:] # Remove dummy 0 in first position
    for i in range(num_five_min_intervals):
        tnow = t0 + datetime.timedelta(minutes=5*i)
        time_list += [tnow.strftime('%Y%m%d%H%M')]
    return time_list, idx_break

def validate(offer, uid):
    top_keys = ['attributes', 'offer']
    attr_keys = ['rid']
    offer_keys = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp', 'block_ch_mc', 'block_dc_mc', 
                  'block_soc_mc', 'block_ch_mq', 'block_dc_mq', 'block_soc_mq', 'soc_end']
    for tk in top_keys:
        otk = offer.keys()
        assert tk in otk, AssertionError(f"Offer is missing required key '{tk}'.")
    for ak in attr_keys:
        oak = offer[top_keys[0]].keys()
        assert ak in oak, AssertionError(f"Offer:attributes is missing required key '{ak}'.")
    for ok in offer_keys:
        ook = offer[top_keys[1]].keys()
        assert ok in ook, AssertionError(f"Offer:offer is missing required key '{ok}'.")
    print(f"Offer appears valid for market clearing interval {uid}")
    
def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s/utils - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)