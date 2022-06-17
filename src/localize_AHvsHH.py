import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import pywarraychannels
import glob
import localization
import path_classification

# Params
set = "office-walls"
architecture = "AH"
link = "up"
samples = 218
figsize = (5, 3.5)
# Power
p_t_dBm = 20            # dBm
# Noise related
T = 15                  # C
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 3e8                 # m/s
# Antennas
N_UE = 4                # Number of UE antennas in each dimension
N_AP = 8                # Number of AP antennas in each dimension
N_RF_UE = N_UE          # Number of UE RF-chains in total
N_RF_AP = N_AP          # Number of AP RF-chains in total
N_M_UE = N_UE           # Number of UE measurements in each dimension
N_M_AP = N_AP           # Number of AP measurements in each dimension
# Other
Q = 64
K_res = 128

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

# Style
Style = {"AH": "dashed", "HH": "solid"}
Colors = {5: [0, 0, 1], 50: [1, 0, 1], 75: [1, 0, 0]}
Names = {"AH": "Analog", "HH": "Hybrid"}

for p in [75, 50, 5]:
    for architecture in ["AH", "HH"]:
        if architecture == "AH":
            N_RF_UE = 1
        elif architecture == "HH":
            N_RF_UE = N_UE
        else:
            print("Not suported architecture")
            raise
        NFrames = []
        Median = []
        for N_M_UE, N_M_AP in [
            [2, 4], [2, 8], [4, 8], [4, 16], [8, 16]
        ]:
            dict_style = "Sector"
            with open(
                "data/{}/AH/paths-dict/single_{}_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
                    set, "MOMP", dict_style, N_UE, N_AP, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
                "r") as f:
                estimations = json.loads(f.read())
            CTime = [dd["CTime"] for dd in estimations]
            Error = []
            for ii_sample, (estimation, ue_pos, ap) in enumerate(zip(estimations, UE_pos, AP_selected)):
                Power_est = np.asarray(estimation['Power'])
                DoA_est = np.asarray(estimation['DoA'])
                DoD_est = np.asarray(estimation['DoD'])
                DDoF_est = np.asarray(estimation['DDoF'])
                pos = localization.localization_single_ap(AP_pos[ap], DoA_est, DDoF_est, classi=path_classification.RX_TXF(DoA_est, DoD_est, th_az=0.12, th_el=0.12), weights=np.power(10, (Power_est-np.min(Power_est))/20))
                if pos is not None:
                    Error.append(np.linalg.norm(pos-np.array(ue_pos)))
                else:
                    Error.append(np.inf)
            # Dump
            Error = np.sort(Error)
            nframes = (N_M_AP*N_M_UE)**2/(N_RF_AP*N_RF_UE)
            if nframes <= 512:
                NFrames.append(nframes)
                Median.append(np.mean(Error[Error < Error[(p*len(Error))//100]]))
        # Plot
        plt.figure("A", figsize=figsize)
        plt.semilogx(NFrames, Median, linestyle=Style[architecture], color=Colors[p], label="{} best {}% mean".format(Names[architecture], p))

# Plots
xticks = [2, 8, 32, 128, 512]
plt.figure("A", figsize=figsize)
plt.xlabel("Number of training frames")
plt.ylabel("Localization error (m)")
#plt.xlim([0, 30])
plt.ylim([0, 2])
plt.xticks(xticks, ["{}".format(xtick) for xtick in xticks])
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("data\\{}\\figures\\LocErr-architecture.png".format(set), bbox_inches='tight')
plt.savefig("data\\{}\\figures\\LocErr-architecture.pdf".format(set), bbox_inches='tight')
plt.show()