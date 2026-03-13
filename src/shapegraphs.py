"""
Generate shapegraphs from files.
"""
import os

from shapegraphs.readers.bassek import generate_shapegraphs_from_files
from shapegraphs import save_shapegraphs


files = os.listdir("data")

pairs = [(m, p) for grp in [
    [f for f in files if mid in f] for mid in {f.rsplit('_', 1)[-1].replace('.xml','') for f in files}
] for m in grp if 'matchinformation' in m for p in grp if 'positions_raw' in p]

results = []
for m, p in pairs:
    result = generate_shapegraphs_from_files(
        match_info_path=os.path.join("data", m),
        position_data_path=os.path.join("data", p),
        verbose=True
    )
    results.append(result)

save_shapegraphs(results, "shapegraphs.pkl")
