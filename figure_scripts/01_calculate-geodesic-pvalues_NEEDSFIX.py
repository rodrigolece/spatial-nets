import os
from pathlib import Path
import numpy as np

from spatial_nets import *
from spatial_nets import utils


data_dir = Path(os.environ["workdirc"]) / "data"
output_dir = Path("output")

approx = True

dmat = utils.load_dmat(data_dir / "UK_geodesic_dmat.npz")
data = utils.load_flows(data_dir / "UK_commute2011.npz")
locs = LocationsDataClass(data, coords=dmat)
N = len(locs)

print("\nComputing DC gravity model\n")
grav = GravityModel(constraint="doubly", verbose=False)
prediction = grav.fit_transform(locs)
doubly = DoublyConstrained(approx_pvalues=approx, verbose=True)
norm_prediction = doubly.fit_transform(locs, prediction)
grav_dc_pvals = doubly.pvalues()
if grav_dc_pvals.N == N:
    grav_dc_pvals.model = "gravity"
    grav_dc_pvals.constraint = "doubly"
    grav_dc_pvals.approx_pvalues = approx
    grav_dc_pvals.save(output_dir / "pvalues_geo_grav_doubly.pkl")
    np.save(output_dir / "model_geo_grav_doubly.npy", norm_prediction)
else:
    print("something went wrong")


print("\nComputing PC radiation model\n")
rad = RadiationModel(finite_correction=False)
prediction = rad.fit_transform(locs)
prod = ProductionConstrained(approx_pvalues=approx, verbose=True)
norm_prediction = prod.fit_transform(locs, prediction)
rad_pc_pvals = prod.pvalues()
if rad_pc_pvals.N == N:
    rad_pc_pvals.model = "gravity"
    rad_pc_pvals.constraint = "doubly"
    rad_pc_pvals.approx_pvalues = approx
    rad_pc_pvals.save(output_dir / "pvalues_geo_rad_prod.pkl")
    np.save(output_dir / "model_geo_rad_prod.npy", norm_prediction)
else:
    print("something went wrong")

print("End of script!\n")
