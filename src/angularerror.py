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
link = "up"
samples = 218
cases = [
    ("OMP", 1),
    ("OMP", 1.3),
    ("OMP", 1.6),
    ("OMP-dT", 2),
    ("OMP-dT", 4),
    ("OMP-dT", 8),
    ("MOMP", 16),
    ("MOMP", 128),
    ("MOMP", 1024)
]
cases = cases[::-1]
figsize = (5, 3.5)
# Power
p_t_dBm = 20            # dBm
# Noise related
T = 15                  # C
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 3e8                 # m/s
# Antennas
N_UE = 3                # Number of UE antennas in each dimension
N_AP = 6                # Number of AP antennas in each dimension
N_RF_UE = N_UE          # Number of UE RF-chains in total
N_RF_AP = N_AP          # Number of AP RF-chains in total
N_M_UE = N_UE           # Number of UE measurements in each dimension
N_M_AP = N_AP           # Number of AP measurements in each dimension
# Other
Q = 64

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

# Label script
def str_label(method, K):
    method_names = {
        "MOMP": "MOMP",
        "OMP-dT": "SWOMP",
        "OMP": "TD-OMP"
        }
    if K == int(K):
        return "{} $K_{}={}$".format(method_names[method], r"{\rm res}", int(K))
    else:
        return "{} $K_{}={:.1f}$".format(method_names[method], r"{\rm res}", K)

# Style
Style = {"OMP": "dashed", "OMP-dT": "dashdot", "MOMP": "solid"}

for method, K_res in cases:
    if dropDoD:
        with open(
            "data/{}/{}/paths-dropDoD/single_{}_{}_{}_{}_{}_{}.json".format(
                set, architecture, method, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
            "r") as f:
            estimations = json.loads(f.read())
    else:
        if method.split("-")[-1] == "dT":
            with open(
                "data/{}/{}/paths-dropT/single_{}_{}_{}_{}_{}_{}.json".format(
                    set, architecture, method.split("-")[0], N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
                "r") as f:
                estimations = json.loads(f.read())
        else:
            with open(
                "data/{}/{}/paths/single_{}_{}_{}_{}_{}_{}.json".format(
                    set, architecture, method, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
                "r") as f:
                estimations = json.loads(f.read())
        CTime = [dd["CTime"] for dd in estimations]
    Error_DoA = []
    if not dropDoD: Error_DoD = []
    for ii_sample, dd in enumerate(estimations):
        DoA_est = dd["DoA"][0]
        if not dropDoD: DoD_est = dd["DoD"][0]
        Error_DoA.append(np.rad2deg(np.arccos(np.dot(DoA[ii_sample], DoA_est))))
        if not dropDoD: Error_DoD.append(np.rad2deg(np.arccos(np.dot(DoD[ii_sample], DoD_est))))
    plt.figure("A", figsize=figsize)
    plt.plot(np.sort(Error_DoA), np.linspace(0, 100, samples), linestyle=Style[method], label=str_label(method, K_res))
    if not dropDoD:
        plt.figure("D", figsize=figsize)
        plt.plot(np.sort(Error_DoD), np.linspace(0, 100, samples), linestyle=Style[method], label=str_label(method, K_res))

# Plots
plt.figure("A", figsize=figsize)
plt.xlabel("DoA error (°)")
plt.ylabel("Cumulative probability [%]")
plt.xlim([0, 30])
plt.ylim([0, 100])
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\DoAErr.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\DoAErr.pdf".format(set, architecture), bbox_inches='tight')

if not dropDoD:
    plt.figure("D", figsize=figsize)
    plt.xlabel("DoD error (°)")
    plt.ylabel("Cumulative probability [%]")
    plt.xlim([0, 30])
    plt.ylim([0, 100])
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("data\\{}\\{}\\figures\\DoDErr.png".format(set, architecture), bbox_inches='tight')
    plt.savefig("data\\{}\\{}\\figures\\DoDErr.pdf".format(set, architecture), bbox_inches='tight')
plt.show()
