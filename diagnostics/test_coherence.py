import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd
from helper_code import load_signal_data, HEADERS
from src.data.features_coherence import standardize_channels as _standardize_for_coherence
from src.data.features_coherence import extract_all_connectivity_features

R='/scratch/simran/pn26/raw/training_set_small'
rows=pd.read_csv(R+'/demographics.csv').to_dict('records')[:3]
for row in rows:
    p=f"{R}/physiological_data/{row['SiteID']}/{row['BidsFolder']}_ses-{row['SessionID']}.edf"
    a=f"{R}/algorithmic_annotations/{row['SiteID']}/{row['BidsFolder']}_ses-{row['SessionID']}_caisr_annotations.edf"
    ch,fs=load_signal_data(p)
    algo,_=load_signal_data(a)
    std,std_fs=_standardize_for_coherence(ch,fs,'channel_table.csv')
    avail=[c for c in ['c3-m2','c4-m1','f3-m2','f4-m1','o1-m2','o2-m1'] if c in std]
    t=time.time()
    f=extract_all_connectivity_features(std,std_fs,algo.get('stage_caisr'))
    ok=sum(1 for v in f.values() if np.isfinite(v))
    print(f"{row['SiteID']} {row['BidsFolder']}: {len(avail)} derivations, "
          f"{len(f)} features, {ok} finite, {time.time()-t:.1f}s")
    for k in ['coh_c3_c4_alpha_n2','coh_f3_o1_delta_n3','coh_mean_sigma_n2','iaf_wake','taf_wake']:
        if k in f: print(f"    {k:26s} {f[k]}")
