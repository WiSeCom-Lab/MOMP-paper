import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import localization
import path_classification

# Params
set = "office-walls"
architecture = "AH"
link = "up"
samples = 218
cases = [
    ("MOMP", 128, -40),
    ("MOMP", 128, -30),
    ("MOMP", 128, -20),
    ("MOMP", 128, -10),
    ("MOMP", 128, 0),
    ("MOMP", 128, 10),
    ("MOMP", 128, 20),
    ("MOMP", 128, 30),
    ("MOMP", 128, 40)
]
figsize = (5, 3.5)
# Power
#p_t_dBm = 20            # dBm
# Noise related
T = 15                  # C
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 3e8
# Antennas
N_UE = 4                # Number of UE antennas in each dimension
N_AP = 8                # Number of AP antennas in each dimension
N_RF_UE = 1             # Number of UE RF-chains in total
N_RF_AP = N_AP          # Number of AP RF-chains in total
# Carriers
f_c = 60                # GHz
B = 2                   # GHz
K = 64                  # Number of frequency carriers
Q = 64
# Measurements
N_M_UE = N_UE              # Number of UE measurements in each dimension
N_M_AP = N_AP              # Number of AP measurements in each dimension
# Estimation
N_est = 5               # Number of estimated paths
K_res_lr = 2            # Coarse resolution factor

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

# Crop data
UE_pos, AP_selected = [X[:samples] for X in [UE_pos, AP_selected]]

# Loop
Pt_vec = []
Best_50_Error_loc = []
Success = []
for method, K_res, p_t_dBm in cases:
    with open(
        "data/{}/{}/paths/single_{}_{}_{}_{}_{}_{}.json".format(
            set, architecture, method, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
        "r") as f:
        estimations = json.loads(f.read())

    # Localize
    UE_pos_est = []
    for estimation, ue_pos, ap, ii_ue in zip(estimations, UE_pos, AP_selected, range(len(UE_pos))):
        Power = np.asarray(estimation['Power'])
        DoA = np.asarray(estimation['DoA'])
        DoD = np.asarray(estimation['DoD'])
        DDoF = np.asarray(estimation['DDoF'])
        classi = path_classification.RX_TXF(DoA, DoD, th_az=0.08, th_el=0.08)
        #pos = localization.localization_eusipco(AP_pos[ap], DoA, DDoF, classi=classi, weights=np.power(10, (Power-np.min(Power))/20))
        pos = localization.localization_single_ap(AP_pos[ap], DoA, DDoF, classi=classi, weights=np.power(10, (Power-np.min(Power))/20))
        #pos = localization.localization_single_ap_NLoS(AP_pos[ap], DoA, DoD, DDoF, weights=np.power(10, (Power-np.min(Power))/20))
        #pos = localization.localization_single_ap_classic(AP_pos[ap], DoA, DoD, DDoF, weights=np.power(10, (Power-np.min(Power))/20))
        if pos is not None:
            UE_pos_est.append(pos.tolist())
        else:
            UE_pos_est.append(None)

    UE_pos_crop = np.asarray([ue_pos for ii_ue, ue_pos in enumerate(UE_pos) if UE_pos_est[ii_ue] is not None])
    UE_pos_est_crop = np.asarray([ue_pos for ue_pos in UE_pos_est if ue_pos is not None])
    success = np.sum([ue_pos is not None for ue_pos in UE_pos_est])
    if success > 1:
        # Statistical information
        AP_sel_crop = np.asarray([AP_pos[ap] for ii_ue, ap in enumerate(AP_selected) if UE_pos_est[ii_ue] is not None])

        ### Location error
        Error_loc = np.linalg.norm(UE_pos_crop-UE_pos_est_crop, ord = 2, axis = 1)
        Error_loc[np.isnan(Error_loc)] = np.Inf

        ### Distance error
        Error_dist = np.abs(np.linalg.norm(UE_pos_crop-AP_sel_crop, ord = 2, axis = 1)-np.linalg.norm(UE_pos_est_crop-AP_sel_crop, ord = 2, axis = 1))
        Error_dist[np.isnan(Error_dist)] = np.Inf

        ### Angle error
        Error_angle = (180/np.pi)*np.arccos(np.sum((UE_pos_crop-AP_sel_crop)*(UE_pos_est_crop-AP_sel_crop), axis = 1)/(np.linalg.norm(UE_pos_crop-AP_sel_crop, ord = 2, axis = 1)*np.linalg.norm(UE_pos_est_crop-AP_sel_crop, ord = 2, axis = 1)))
        Error_loc_angle = Error_angle*(np.pi/180)*np.linalg.norm(UE_pos_crop-AP_sel_crop, ord = 2, axis = 1)
        Error_angle[np.isnan(Error_angle)] = np.Inf
        Error_loc_angle[np.isnan(Error_loc_angle)] = np.Inf

        # Plot
        plt.figure("L", figsize=figsize)
        plt.plot(np.sort(Error_loc), np.linspace(0, 100*success/samples, success), label="{}dB".format(p_t_dBm))
        plt.figure("A", figsize=figsize)
        plt.plot(np.sort(Error_loc_angle), np.linspace(0, 100*success/samples, success), label="{}dB".format(p_t_dBm))
        plt.figure("R", figsize=figsize)
        plt.plot(np.sort(Error_dist), np.linspace(0, 100*success/samples, success), label="{}dB".format(p_t_dBm))
        #print("For {} with K_res={}, we achieve an error below {}m for 50% of the users, of which {}m correspond to the angle induced error and {}m to the ranging induced one".format(method, K_res, np.quantile(Error_loc, 0.5*samples/success), np.quantile(Error_loc_angle, 0.5*samples/success), np.quantile(Error_dist, 0.5*samples/success)))

        # Collect vector information
        Pt_vec.append(p_t_dBm)
        #Best_50_Error_loc.append([err for err in Error_loc if err < np.quantile(Error_loc, 0.5*samples/success)])
        Best_50_Error_loc.append(Error_loc)
        Success.append(success)
    else:
        # Plot
        plt.figure("L", figsize=figsize)
        plt.plot([0, 100], [0, 0], label="{}dB".format(p_t_dBm))
        plt.figure("A", figsize=figsize)
        plt.plot([0, 100], [0, 0], label="{}dB".format(p_t_dBm))
        plt.figure("R", figsize=figsize)
        plt.plot([0, 100], [0, 0], label="{}dB".format(p_t_dBm))

# Build plots
plt.figure("L", figsize=figsize)
plt.xlim([0, 10])
plt.ylim([0, 100])
plt.xlabel("Localization error [m]")
plt.ylabel("Cumulative probability [%]")
plt.legend()
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\LocErr.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\LocErr.pdf".format(set, architecture), bbox_inches='tight')
plt.figure("A", figsize=figsize)
plt.xlim([0, 1])
plt.ylim([0, 100])
plt.xlabel("Localization angle induced error [m]")
plt.ylabel("Cumulative probability [%]")
plt.legend()
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\AngLocErr.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\AngLocErr.pdf".format(set, architecture), bbox_inches='tight')
plt.figure("R", figsize=figsize)
plt.xlim([0, 10])
plt.ylim([0, 100])
plt.xlabel("Localization ranging induced error [m]")
plt.ylabel("Cumulative probability [%]")
plt.legend()
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\RangLocErr.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\RangLocErr.pdf".format(set, architecture), bbox_inches='tight')

# Build statistical plot
Best_50_Error_loc = Best_50_Error_loc
plt.figure("Line", figsize=figsize)
plt.plot([np.quantile(err_vec, 0.75) for err_vec in Best_50_Error_loc], "--", color=cm.get_cmap('Set1')(0), label="p-75")
plt.plot([np.mean(np.sort(err_vec)[:int(0.75*success)]) for err_vec, success in zip(Best_50_Error_loc, Success)], "-", color=cm.get_cmap('Set1')(0), label="Best 75% mean")
plt.plot([np.quantile(err_vec, 0.5) for err_vec in Best_50_Error_loc], "--", color=cm.get_cmap('Set1')(1), label="p-50")
plt.plot([np.mean(np.sort(err_vec)[:int(0.5*success)]) for err_vec, success in zip(Best_50_Error_loc, Success)], "-", color=cm.get_cmap('Set1')(1), label="Best 50% mean")
plt.plot([np.quantile(err_vec, 0.05) for err_vec in Best_50_Error_loc], "--", color=cm.get_cmap('Set1')(2), label="p-5")
plt.plot([np.mean([err for err in err_vec if err < np.quantile(err_vec, 0.05)]) for err_vec in Best_50_Error_loc], "-", color=cm.get_cmap('Set1')(2), label="Best 5% mean")
plt.plot([np.min(err_vec) for err_vec in Best_50_Error_loc], "k--", label="min")
plt.xticks(np.arange(len(Pt_vec)), Pt_vec)
plt.xlabel("Transmitted power [dBm]")
#plt.yscale("log")
plt.ylim([0, 3])
plt.ylabel("Localization error [m]")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\LineLocErr_Pt.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\LineLocErr_Pt.pdf".format(set, architecture), bbox_inches='tight')
plt.figure("Detection", figsize=figsize)
plt.plot([100*success/samples for success in Success], "k-")
plt.xticks(np.arange(len(Pt_vec)), Pt_vec)
plt.xlabel("Transmitted power [dBm]")
#plt.yscale("log")
plt.ylim([0, 100])
plt.ylabel("Detection probability [%]")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data\\{}\\{}\\figures\\LineLocErr_Pt_s.png".format(set, architecture), bbox_inches='tight')
plt.savefig("data\\{}\\{}\\figures\\LineLocErr_Pt_s.pdf".format(set, architecture), bbox_inches='tight')
plt.show()
