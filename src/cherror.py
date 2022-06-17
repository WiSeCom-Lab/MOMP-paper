import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import localization
import path_classification
import pywarraychannels

# Params
set = "office-walls"
architecture = "AH"
link = "up"
samples = 218
cases = [
    ("OMP", 1),
    ("OMP", 1.3),
    ("OMP", 1.6),
    ("MOMP", 16),
    ("MOMP", 128),
    ("MOMP", 1024)
]
cases = cases[::-1]
figsize = (5, 3.5)
min_snr_th = None
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
orientations_AP = [pywarraychannels.uncertainties.Static(roll=np.pi/2)]*6 + [pywarraychannels.uncertainties.Static(roll=-np.pi/2)]*6
# Carriers
f_c = 60                # GHz
B = 2                   # GHz
K = 64                  # Number of delay taps
Q = 64                  # Length of the training pilot
# Define antennas
antenna_UE = pywarraychannels.antennas.RectangularAntenna((N_UE, N_UE))
antenna_AP = pywarraychannels.antennas.RectangularAntenna((N_AP, N_AP))
# Transform params to natural units
f_c *= 1e9
B *= 1e9
T += 273.1
p_t = np.power(10, (p_t_dBm-30)/10)
p_n = k_B*T*B

# Define pulse shape filter
filter = pywarraychannels.filters.RCFilter(early_samples=8, late_samples=8)
# Define antennas
antenna_UE = pywarraychannels.antennas.RectangularAntenna((N_UE, N_UE))
antenna_AP = pywarraychannels.antennas.RectangularAntenna((N_AP, N_AP))
# Define channel Geomtric
if link == "up":
    channel_Geometric = pywarraychannels.channels.Geometric(
        antenna_AP, antenna_UE, f_c=f_c,
        B=B, K=K, filter=filter, bool_sync=True)
else:
    channel_Geometric = pywarraychannels.channels.Geometric(
        antenna_UE, antenna_AP, f_c=f_c,
        B=B, K=K, filter=filter, bool_sync=True)

# Initialization
Quartiles_loc = []
Quartiles_dist = []
Quartiles_angle = []
Quartiles_loc_angle = []

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

# Filter by SNR
SNR = np.asarray([(np.max(rays.ray_info[:, 2])-30)+(p_t_dBm-30)-(10*np.log10(p_n)) for rays in Rays])
if min_snr_th is not None:
    UE_pos, AP_selected = [np.asarray(x)[SNR > min_snr_th] for x in [UE_pos, AP_selected]]
    samples = len(UE_pos)

# Label script
def str_label(method, K):
    method_names = {
        "MOMP": "MOMP",
        "OMP-dT": "SWOMP",
        "OMP": "TD-OMP"
        }
    if K == int(K):
        return "{} ${}$".format(method_names[method], int(K))
    else:
        return "{} ${:.1f}$".format(method_names[method], K)

# Style
Style = {"OMP": "--", "MOMP": "-"}

# Loop
plt.figure(figsize=figsize)
for method, K_res in cases:
    with open(
        "data/{}/{}/paths/single_{}_{}_{}_{}_{}_{}.json".format(
            set, architecture, method, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
        "r") as f:
        estimations = json.loads(f.read())

    # Filter SNR
    if min_snr_th is not None:
        estimations = [est for est, snr in zip(estimations, SNR) if snr > min_snr_th]

    # Localize
    channel_err = []
    for estimation, rays, ap, ii_ue in zip(estimations, Rays, AP_selected, range(len(UE_pos))):
        antenna_AP.uncertainty = orientations_AP[ap]
        #antenna_UE.uncertainty = orientations_AP[ap]
        Alpha = np.asarray(estimation['Alpha_r'])+1j*np.asarray(estimation['Alpha_i'])
        Power = np.asarray(estimation['Power'])
        DoA = np.asarray(estimation['DoA'])
        DoD = np.asarray(estimation['DoD'])
        DDoF = np.asarray(estimation['DDoF'])
        Phase = np.rad2deg(np.angle(Alpha))
        TDoF = DDoF/c
        if method == "OMP":
            Power_dBm = Power-(p_t_dBm-30)+30-10*np.log10(Q)
        else:
            Power_dBm = Power-(p_t_dBm-30)+30-10*np.log10(Q)
        DoA_azel = np.asarray([pywarraychannels.em.cartesian2polar(doa) for doa in DoA])
        DoD_azel = np.asarray([pywarraychannels.em.cartesian2polar(dod) for dod in DoD])
        Angles = np.rad2deg(np.concatenate([DoA_azel, DoD_azel], axis=1))
        rays_info = np.concatenate([Phase[:, np.newaxis], TDoF[:, np.newaxis], Power_dBm[:, np.newaxis], Angles], axis=1)
        rays_est = pywarraychannels.em.Geometric(rays_info)
        ch = channel_Geometric.build(rays)
        ch_est = channel_Geometric.build(rays_est)
        ch_err = np.linalg.norm((ch-ch_est).reshape([-1]), ord=2, axis=0)/np.linalg.norm(ch.reshape([-1]), ord=2, axis=0)
        channel_err.append(ch_err)
        if ii_ue == 0:
            print(method, K_res)
            print(rays)
            print(rays_est)

    # Plot
    plt.plot(20*np.log10(np.sort(channel_err)), np.linspace(0, 100, samples), Style[method], label=str_label(method, K_res))

# Build plots
#plt.xlim([-25, 5])
plt.ylim([0, 100])
plt.xlabel("Channel NMSE [dB]")
plt.ylabel("Cumulative probability [%]")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\ChErr.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\ChErr.pdf".format(set, architecture), bbox_inches='tight')
plt.show()
