"""Write out the exact feature list the submission trains and predicts on.

The list has to be identical in both, and it is derived rather than typed: the
inline extractor's own output keys, intersected with what the offline matrix
carries, plus the temporal block and recording year. Deriving it means a change
to the extractor cannot silently desynchronize the two.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
from src.data.feature_io import load_features
import team_code
from helper_code import find_patients

R='/scratch/simran/pn26/raw/training_set_small'
inline=set(team_code._extract_one(R, find_patients(R+'/demographics.csv')[0],
                                  R+'/demographics.csv','channel_table.csv')[0].keys())
df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
tp=[c for c in cols if c.startswith('tp_')]
ship=sorted(set(cols)&(inline|set(tp)|{'rec_year'}))
out='configs/shipped_features.json'
json.dump(ship, open(out,'w'), indent=1)
print(f'{len(ship)} features -> {out}')
print(f'  inline {len(set(ship)&inline)}, temporal {len([c for c in ship if c.startswith("tp_")])}, '
      f'rec_year {"rec_year" in ship}')
