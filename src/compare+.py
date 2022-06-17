import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import pywarraychannels
import glob

# Params
set = "office-walls"
architecture = "AH"
dropDoD = False
samples = 218
link = "up"
figsize = (5, 3.5)

# Load data
with open("data/{}/AP_pos.txt".format(set)) as f:
    AP_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/{}/UE_pos.txt".format(set)) as f:
    UE_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/{}/AP_selected.txt".format(set)) as f:
    AP_selected = [int(a) for a in f.read().split("\n")[1].split()]
with open("data/{}/Info_selected.txt".format(set)) as f:
    Rays = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=link=="up") for ue_block in f.read()[:-1].split("\n<ue>\n")]

# Crop data
UE_pos, AP_selected, Rays = [X[:samples] for X in [UE_pos, AP_selected, Rays]]

# Preprocess data
DoA_az = [np.deg2rad(rays.ray_info[0, 3]) for rays in Rays]
DoA_el = [np.deg2rad(rays.ray_info[0, 4]) for rays in Rays]
DoD_az = [np.deg2rad(rays.ray_info[0, 5]) for rays in Rays]
DoD_el = [np.deg2rad(rays.ray_info[0, 6]) for rays in Rays]
DoA = np.asarray([pywarraychannels.em.polar2cartesian(doa_az, doa_el) for doa_az, doa_el in zip(DoA_az, DoA_el)])
DoD = np.asarray([pywarraychannels.em.polar2cartesian(dod_az, dod_el) for dod_az, dod_el in zip(DoD_az, DoD_el)])
print("LoS scenarios: {:.1f}%".format(100*np.mean(np.linalg.norm(DoA+DoD, ord=2, axis=1) < 0.01)))

# Plot data init
OMP_K = []
OMP_CTime_median = []
OMP_Error_DoA_median = []
OMP_Error_DoD_median = []

OMP_dT_K = []
OMP_dT_CTime_median = []
OMP_dT_Error_DoA_median = []
OMP_dT_Error_DoD_median = []

MOMP_K = []
MOMP_CTime_median = []
MOMP_Error_DoA_median = []
MOMP_Error_DoD_median = []

# Cicle over files
files = [[file, "RX-TX-T"] for file in glob.glob("data\\{}\\{}\\paths\\*.json".format(set, architecture))]+[[file, "RX-TX"] for file in glob.glob("data\\{}\\{}\\paths-dropT\\*.json".format(set, architecture))]
for file, kind in files:
    with open(file, "r") as f: data = json.loads(f.read())
    method, N_M_UE, N_M_AP, Q, p_t_dBm, K10 = file.split("\\")[-1].split(".")[0].split("_")[1:]
    N_M_UE = int(N_M_UE)
    N_M_AP = int(N_M_AP)
    p_t_dBm = int(p_t_dBm)
    Q = int(Q)
    K = int(K10)/10
    if not (N_M_UE == 3 and N_M_AP == 6 and p_t_dBm == 20): continue
    if file.split("\\")[-2] == "paths-dropT" and method == "MOMP": continue
    if file.split("\\")[-2] == "paths-dropT" and K == 1: continue
    CTime = [dd["CTime"] for dd in data]
    Error_DoA = []
    if not dropDoD: Error_DoD = []
    for ii_sample, dd in enumerate(data):
        DoA_est = dd["DoA"][0]
        if not dropDoD: DoD_est = dd["DoD"][0]
        Error_DoA.append(np.rad2deg(np.arccos(np.dot(DoA[ii_sample], DoA_est))))
        if not dropDoD: Error_DoD.append(np.rad2deg(np.arccos(np.dot(DoD[ii_sample], DoD_est))))
        """
        print(Rays[ii_sample].first(1))
        print([np.rad2deg(a) for a in pywarraychannels.em.cartesian2polar(DoA_est)], [np.rad2deg(a) for a in pywarraychannels.em.cartesian2polar(DoD_est)])
        if method == "OMP" and K == 13: raise
        """
    if method == "OMP":
        if kind == "RX-TX-T":
            OMP_K.append(K)
            OMP_CTime_median.append(np.median(CTime))
            OMP_Error_DoA_median.append(np.median(Error_DoA))
            if not dropDoD: OMP_Error_DoD_median.append(np.median(Error_DoD))
        if kind == "RX-TX":
            OMP_dT_K.append(K)
            OMP_dT_CTime_median.append(np.median(CTime))
            OMP_dT_Error_DoA_median.append(np.median(Error_DoA))
            if not dropDoD: OMP_dT_Error_DoD_median.append(np.median(Error_DoD))
    elif method == "MOMP":
        MOMP_K.append(K)
        MOMP_CTime_median.append(np.median(CTime))
        MOMP_Error_DoA_median.append(np.median(Error_DoA))
        if not dropDoD: MOMP_Error_DoD_median.append(np.median(Error_DoD))

# Plots
plt.figure("TvDoA", figsize=figsize)
plt.loglog(OMP_CTime_median, OMP_Error_DoA_median, "o", label="TD-OMP")
plt.loglog(OMP_dT_CTime_median, OMP_dT_Error_DoA_median, "o", label="SWOMP")
plt.loglog(MOMP_CTime_median, MOMP_Error_DoA_median, "o", label="MOMP")
plt.xlabel("Computation time [s]")
plt.ylabel("DoA error (°)")
plt.legend()
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\Time_vs_DoAE.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\Time_vs_DoAE.pdf".format(set, architecture), bbox_inches='tight')

if not dropDoD:
    plt.figure("TvDoD", figsize=figsize)
    plt.loglog(OMP_CTime_median, OMP_Error_DoD_median, "o", label="TD-OMP")
    plt.loglog(OMP_dT_CTime_median, OMP_dT_Error_DoD_median, "o", label="SWOMP")
    plt.loglog(MOMP_CTime_median, MOMP_Error_DoD_median, "o", label="MOMP")
    plt.xlabel("Computation time [s]")
    plt.ylabel("DoD error (°)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data\\{}\\{}\\figures\\Time_vs_DoDE.png".format(set, architecture), bbox_inches='tight')
    plt.savefig("data\\{}\\{}\\figures\\Time_vs_DoDE.pdf".format(set, architecture), bbox_inches='tight')

plt.figure("KvDoA", figsize=figsize)
plt.loglog(OMP_K, OMP_Error_DoA_median, "o", label="TD-OMP")
plt.loglog(OMP_dT_K, OMP_dT_Error_DoA_median, "o", label="SWOMP")
plt.loglog(MOMP_K, MOMP_Error_DoA_median, "o", label="MOMP")
MOMP_off_x = [0.6, 0.8, 0.8]
MOMP_off_y = [1.1, 1.1, 1.1]
OMP_off_x = [1.15, 1.15, 1.15]
OMP_off_y = [0.96, 0.9, 0.85]
OMP_dT_off_x = [1.15, 1.15, 1.15]
OMP_dT_off_y = [0.96, 0.9, 1]
for ii_MOMP in range(len(MOMP_CTime_median)):
    plt.text(MOMP_K[ii_MOMP]*MOMP_off_x[ii_MOMP%len(MOMP_off_x)], MOMP_Error_DoA_median[ii_MOMP]*MOMP_off_y[ii_MOMP%len(MOMP_off_y)], "{:.1f}s".format(MOMP_CTime_median[ii_MOMP]))
for ii_OMP in range(len(OMP_CTime_median)):
    plt.text(OMP_K[ii_OMP]*OMP_off_x[ii_OMP%len(OMP_off_x)], OMP_Error_DoA_median[ii_OMP]*OMP_off_y[ii_OMP%len(OMP_off_y)], "{:.1f}s".format(OMP_CTime_median[ii_OMP]))
for ii_OMP_dT in range(len(OMP_dT_CTime_median)):
    plt.text(OMP_dT_K[ii_OMP_dT]*OMP_dT_off_x[ii_OMP_dT%len(OMP_dT_off_x)], OMP_dT_Error_DoA_median[ii_OMP_dT]*OMP_dT_off_y[ii_OMP_dT%len(OMP_dT_off_y)], "{:.1f}s".format(OMP_dT_CTime_median[ii_OMP_dT]))
plt.xlabel(r"$K_{\rm res}$")
plt.ylabel("DoA error (°)")
plt.legend()
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\K_vs_DoAE.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\K_vs_DoAE.pdf".format(set, architecture), bbox_inches='tight')

if not dropDoD:
    plt.figure("KvDoD", figsize=figsize)
    print("MOMP")
    [print(K, EA, ED, T) for K, EA, ED, T in zip(
        MOMP_K, MOMP_Error_DoA_median, MOMP_Error_DoD_median, MOMP_CTime_median
    )]
    print("OMP")
    [print(K, EA, ED, T) for K, EA, ED, T in zip(
        OMP_K, OMP_Error_DoA_median, OMP_Error_DoD_median, OMP_CTime_median
    )]
    print("OMP-dT")
    [print(K, EA, ED, T) for K, EA, ED, T in zip(
        OMP_dT_K, OMP_dT_Error_DoA_median, OMP_dT_Error_DoD_median, OMP_dT_CTime_median
    )]
    plt.loglog(OMP_K, OMP_Error_DoD_median, "o", label="TD-OMP")
    plt.loglog(OMP_dT_K, OMP_dT_Error_DoD_median, "o", label="SWOMP")
    plt.loglog(MOMP_K, MOMP_Error_DoD_median, "o", label="MOMP")
    MOMP_off_x = [0.6, 0.8, 1.1]
    MOMP_off_y = [1.1, 1.1, 0.8]
    OMP_off_x = [1.15, 1.15, 0.7]
    OMP_off_y = [0.96, 0.95, 1.1]
    OMP_dT_off_x = [1.1, 1.15, 0.7]
    OMP_dT_off_y = [0.8, 0.95, 1.1]
    for ii_MOMP in range(len(MOMP_CTime_median)):
        plt.text(MOMP_K[ii_MOMP]*MOMP_off_x[ii_MOMP%len(MOMP_off_x)], MOMP_Error_DoD_median[ii_MOMP]*MOMP_off_y[ii_MOMP%len(MOMP_off_y)], "{:.1f}s".format(MOMP_CTime_median[ii_MOMP]))
    for ii_OMP in range(len(OMP_CTime_median)):
        plt.text(OMP_K[ii_OMP]*OMP_off_x[ii_OMP%len(OMP_off_x)], OMP_Error_DoD_median[ii_OMP]*OMP_off_y[ii_OMP%len(OMP_off_y)], "{:.1f}s".format(OMP_CTime_median[ii_OMP]))
    for ii_OMP_dT in range(len(OMP_dT_CTime_median)):
        plt.text(OMP_dT_K[ii_OMP_dT]*OMP_dT_off_x[ii_OMP_dT%len(OMP_dT_off_x)], OMP_dT_Error_DoD_median[ii_OMP_dT]*OMP_dT_off_y[ii_OMP_dT%len(OMP_dT_off_y)], "{:.1f}s".format(OMP_dT_CTime_median[ii_OMP_dT]))
    plt.xlabel(r"$K_{\rm res}$")
    plt.ylabel("DoD error (°)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data\\{}\\{}\\figures\\K_vs_DoDE.png".format(set, architecture), bbox_inches='tight')
    plt.savefig("data\\{}\\{}\\figures\\K_vs_DoDE.pdf".format(set, architecture), bbox_inches='tight')

plt.figure("KvT", figsize=figsize)
plt.loglog(OMP_K, OMP_CTime_median, "o", label="TD-OMP")
plt.loglog(OMP_dT_K, OMP_dT_CTime_median, "o", label="SWOMP")
plt.loglog(MOMP_K, MOMP_CTime_median, "o", label="MOMP")
plt.xlabel(r"$K_{\rm res}$")
plt.ylabel("Computation time [s]")
plt.legend()
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\K_vs_T.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\K_vs_T.pdf".format(set, architecture), bbox_inches='tight')
plt.show()
