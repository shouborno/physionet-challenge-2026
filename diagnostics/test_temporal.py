import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd
from helper_code import load_signal_data
from src.data.features_coherence import standardize_channels
from src.data.features_temporal import extract_temporal_pooling_features
R='/scratch/simran/pn26/raw/training_set_small'
for row in pd.read_csv(R+'/demographics.csv').to_dict('records')[:3]:
    p=f"{R}/physiological_data/{row['SiteID']}/{row['BidsFolder']}_ses-{row['SessionID']}.edf"
    a=f"{R}/algorithmic_annotations/{row['SiteID']}/{row['BidsFolder']}_ses-{row['SessionID']}_caisr_annotations.edf"
    ch,fs=load_signal_data(p); algo,_=load_signal_data(a)
    std,sfs=standardize_channels(ch,fs,'channel_table.csv')
    t=time.time(); f=extract_temporal_pooling_features(std,sfs,algo.get('stage_caisr'))
    ok=sum(1 for v in f.values() if np.isfinite(v))
    print(f"{row['SiteID']}: {len(f)} features, {ok} finite, {time.time()-t:.1f}s")
    for k in ['tp_delta_q88','tp_delta_q50','tp_delta_cv','tp_alpha_worst_hour_frac','tp_ratio_delta_alpha_q88','tp_theta_drift']:
        if k in f: print(f'    {k:30s} {f[k]:.5f}')
