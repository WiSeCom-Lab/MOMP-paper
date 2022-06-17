import pywarraychannels
import numpy as np
import json
import scipy
import MOMP
from time import time

# Loop
cases = [
    #("MOMP", "Sector", 128, 4, 8, 2, 4, 64, 20),
    #("MOMP", "Sector", 128, 4, 8, 2, 8, 64, 20),
    #("MOMP", "Sector", 128, 4, 8, 4, 8, 64, 20),
    #("MOMP", "Sector", 128, 4, 8, 4, 16, 64, 20),
    #("MOMP", "Sector", 128, 4, 8, 8, 16, 64, 20),
    ("MOMP", "Random", 128, 4, 8, 2, 4, 64, 20),
    ("MOMP", "Random", 128, 4, 8, 2, 8, 64, 20),
    ("MOMP", "Random", 128, 4, 8, 4, 8, 64, 20),
    ("MOMP", "Random", 128, 4, 8, 4, 16, 64, 20),
    ("MOMP", "Random", 128, 4, 8, 8, 16, 64, 20),
    ("MOMP", "Ge2022", 128, 4, 8, 2, 4, 64, 20),
    ("MOMP", "Ge2022", 128, 4, 8, 2, 8, 64, 20),
    ("MOMP", "Ge2022", 128, 4, 8, 4, 8, 64, 20),
    ("MOMP", "Ge2022", 128, 4, 8, 4, 16, 64, 20),
    ("MOMP", "Ge2022", 128, 4, 8, 8, 16, 64, 20),
]
for method, dict_style, K_res, N_UE, N_AP, N_M_UE, N_M_AP, Q, p_t_dBm in cases:
    # Params
    set = "office-walls"         # Dataset
    link = "up"             # Whether it's up-link or down-link
    #method = "MOMP"         # Channel estimation method (MOMP or OMP)
    #K_res = 128             # Method's dictionary resolution
    K_res_lr = 4            # Method's dictionary low resolution
    samples = 218           # Number of samples from the dataset to evaluate
    # Power
    #p_t_dBm = 20            # dBm
    # Noise related
    T = 15                  # C
    k_B = 1.38064852e-23    # Boltzmanz's constant
    # Speed of light
    c = 3e8                 # m/s
    # Antennas
    #N_UE = 4                # Number of UE antennas in each dimension
    #N_AP = 8                # Number of AP antennas in each dimension
    N_RF_UE = 1              # Number of UE RF-chains in total
    N_RF_AP = N_AP           # Number of AP RF-chains in total
    #N_M_UE = N_UE            # Number of UE measurements in each dimension
    #N_M_AP = N_AP            # Number of AP measurements in each dimension
    if set == "office-corners":
        orientations_AP = [pywarraychannels.uncertainties.Static(pan=np.pi/4, tilt=-2*np.pi/3), pywarraychannels.uncertainties.Static(pan=5*np.pi/4, tilt=-np.pi/2)]
    elif set == "office-walls":
        orientations_AP = [pywarraychannels.uncertainties.Static(tilt=np.pi/2), pywarraychannels.uncertainties.Static(tilt=-np.pi/2)]
    # Carriers
    f_c = 60                # GHz
    B = 2                   # GHz
    K = 64                  # Number of delay taps
    #Q = 64                  # Length of the training pilot
    # Estimation
    N_est = 5               # Number of estimated paths

    # Define pulse shape filter
    filter = pywarraychannels.filters.RCFilter(early_samples=8, late_samples=8)

    # Pilot signals
    if link == "up":
        Pilot = np.concatenate([scipy.linalg.hadamard(Q)[:N_RF_UE], np.zeros((N_RF_UE, K//2))], axis=1)/np.sqrt(N_RF_UE)
    else:
        Pilot = np.concatenate([scipy.linalg.hadamard(Q)[:N_RF_AP], np.zeros((N_RF_AP, K//2))], axis=1)/np.sqrt(N_RF_AP)
    P_len = Pilot.shape[1]
    D = K+filter.early_samples+filter.late_samples
    Pilot = np.concatenate([np.zeros((Pilot.shape[0], D)), Pilot], axis=1)          # Zero-padding

    # Define antennas
    antenna_UE = pywarraychannels.antennas.RectangularAntenna((N_UE, N_UE))
    antenna_AP = pywarraychannels.antennas.RectangularAntenna((N_AP, N_AP))

    # Define codebooks
    if dict_style == "Sector":
        antenna_UE.set_reduced_codebook((N_M_UE, N_M_UE))
        antenna_AP.set_reduced_codebook((N_M_AP, N_M_AP))
    elif dict_style == "Random":
        phases_UE = 2*np.pi*np.random.rand(N_UE**2, N_M_UE**2)
        phases_AP = 2*np.pi*np.random.rand(N_AP**2, N_M_AP**2)
        antenna_UE.set_codebook(np.cos(phases_UE)+1j*np.sin(phases_UE))
        antenna_AP.set_codebook(np.cos(phases_AP)+1j*np.sin(phases_AP))
    elif dict_style == "Ge2022":
        from scipy.io import loadmat
        loaded_cdbks = loadmat("./src/Ge2022-dicts/Ge2022_AH_{}_{}_{}_{}.mat".format(N_UE, N_AP, N_M_UE, N_M_AP))
        antenna_UE.set_codebook(loaded_cdbks["Dict_t"])
        antenna_AP.set_codebook(loaded_cdbks["Dict_r"])
    else:
        print("dict_style {} not defined".format(dict_style))
        raise

    # Split codebooks according to number of RF-chains
    cdbks_UE = np.transpose(np.reshape(antenna_UE.codebook, [N_UE**2, -1, N_RF_UE]), [1, 0, 2])
    cdbks_AP = np.transpose(np.reshape(antenna_AP.codebook, [N_AP**2, -1, N_RF_AP]), [1, 0, 2])

    # Transform params to natural units
    f_c *= 1e9
    B *= 1e9
    T += 273.1
    p_t = np.power(10, (p_t_dBm-30)/10)

    # Compute noise level
    p_n = k_B*T*B
    print("Noise level: {:.2f}dBm".format(10*np.log10(p_n)+30))

    # Define channel Geomtric-MIMO-AWGN
    if link == "up":
        channel_Geometric = pywarraychannels.channels.Geometric(
            antenna_AP, antenna_UE, f_c=f_c,
            B=B, K=K, filter=filter, bool_sync=True)
    else:
        channel_Geometric = pywarraychannels.channels.Geometric(
            antenna_UE, antenna_AP, f_c=f_c,
            B=B, K=K, filter=filter, bool_sync=True)
    channel_MIMO = pywarraychannels.channels.MIMO(channel_Geometric, pilot=Pilot)
    channel = pywarraychannels.channels.AWGN(channel_MIMO, power=p_t, noise=p_n)
    print("Channel")

    # Whitening matrices
    if link == "up":
        LLinv = [np.linalg.inv(np.linalg.cholesky(np.dot(np.conj(cdbk.T), cdbk))) for cdbk in cdbks_AP]
    else:
        LLinv = [np.linalg.inv(np.linalg.cholesky(np.dot(np.conj(cdbk.T), cdbk))) for cdbk in cdbks_UE]

    # Measurement matrix
    if link == "up":
        L_invW = []
        for cdbk_AP, Linv in zip(cdbks_AP, LLinv):
            L_invW.append(np.dot(Linv, np.conj(cdbk_AP.T)))
        L_invW = np.concatenate(L_invW, axis=0)     # N_M_RX  x  N_RX
        F = []
        for cdbk_UE in cdbks_UE:
            F.append(cdbk_UE.T)
        F= np.concatenate(F, axis=0)               # N_M_RX  x  N_RX
        E_conv = np.zeros((P_len, D))
        for k in range(D):
            E_conv[:, k] = Pilot[0, D-k:P_len+D-k]

        L_invW_U, _, _ = np.linalg.svd(L_invW, full_matrices=False)
        F_U, _, _, = np.linalg.svd(F, full_matrices=False)
        E_conv_U, _, _, = np.linalg.svd(E_conv, full_matrices=False)

        L_invW_x_U = np.tensordot(L_invW_U.conj(), L_invW, axes=(0, 0))
        F_x_U = np.tensordot(F_U.conj(), F, axes=(0, 0))
        E_conv_x_U = np.tensordot(E_conv_U.conj(), E_conv, axes=(0, 0))

        A = L_invW_x_U.reshape((-1, 1, 1, N_AP, N_AP, 1, 1, 1)) * F_x_U.reshape((1, -1, 1, 1, 1, N_UE, N_UE, 1)) * E_conv_x_U.reshape((1, 1, -1, 1, 1, 1, 1, D))
        A = A.reshape((-1, N_AP, N_AP, N_UE, N_UE, D))
    else:
        L_invW = []
        for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
            L_invW.append(np.dot(Linv, np.conj(cdbk_UE.T)))
        L_invW = np.concatenate(L_invW, axis=0)     # N_M_RX  x  N_RX
        F = []
        for cdbk_AP in cdbks_AP:
            F.append(cdbk_AP.T)
        F= np.concatenate(F, axis=0)               # N_M_RX  x  N_RX
        E_conv = np.zeros((P_len, D))
        for k in range(D):
            E_conv[:, k] = Pilot[0, D-k:P_len+D-k]

        L_invW_U, _, _ = np.linalg.svd(L_invW, full_matrices=False)
        F_U, _, _, = np.linalg.svd(F, full_matrices=False)
        E_conv_U, _, _, = np.linalg.svd(E_conv, full_matrices=False)

        L_invW_x_U = np.tensordot(L_invW_U.conj(), L_invW, axes=(0, 0))
        F_x_U = np.tensordot(F_U.conj(), F, axes=(0, 0))
        E_conv_x_U = np.tensordot(E_conv_U.conj(), E_conv, axes=(0, 0))

        A = L_invW_x_U.reshape((-1, 1, 1, N_UE, N_UE, 1, 1, 1)) * F_x_U.reshape((1, -1, 1, 1, 1, N_AP, N_AP, 1)) * E_conv_x_U.reshape((1, 1, -1, 1, 1, 1, 1, D))
        A = A.reshape((-1, N_UE, N_UE, N_AP, N_AP, D))

    # Sparse decomposition components
    angles_AP = np.linspace(-np.pi, np.pi, int(N_M_AP*K_res))
    angles_UE = np.linspace(-np.pi, np.pi, int(N_M_UE*K_res))
    A_AP = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP[np.newaxis, :])
    A_UE = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE[np.newaxis, :])
    delays = np.linspace(0, K, int(K*K_res))
    A_time = filter.response(K, delays)

    # Sparse decomposition components
    angles_AP_lr = np.linspace(-np.pi, np.pi, int(N_M_AP*K_res_lr))
    angles_UE_lr = np.linspace(-np.pi, np.pi, int(N_M_UE*K_res_lr))
    A_AP_lr = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP_lr[np.newaxis, :])
    A_UE_lr = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE_lr[np.newaxis, :])
    delays_lr = np.linspace(0, K, int(K*K_res_lr))
    A_time_lr = filter.response(K, delays_lr)

    # Dictionaries
    if link == "up":
        X = [
            A_AP,
            A_AP,
            np.conj(A_UE),
            np.conj(A_UE),
            A_time
        ]
        X_lr = [
            A_AP_lr,
            A_AP_lr,
            np.conj(A_UE_lr),
            np.conj(A_UE_lr),
            A_time_lr
        ]
    else:
        X = [
            A_UE,
            A_UE,
            np.conj(A_AP),
            np.conj(A_AP),
            A_time
        ]
        X_lr = [
            A_UE_lr,
            A_UE_lr,
            np.conj(A_AP_lr),
            np.conj(A_AP_lr),
            A_time_lr
        ]

    # Define decomposition algorithm
    stop = MOMP.stop.General(maxIter=N_est)     # Stop when reached the desired number of estimated paths
    if method == "OMP":
        X_kron = A.copy()
        for x in X:
            X_kron = np.tensordot(X_kron, x, axes = (1, 0))
        X_kron = np.reshape(X_kron, [X_kron.shape[0], -1])
        proj = MOMP.proj.OMP_proj(X_kron)
    else:
        proj_init = MOMP.proj.MOMP_greedy_proj(A, X, X_lr, normallized=False)
        proj = MOMP.proj.MOMP_proj(A, X, initial=proj_init, normallized=False)
    alg = MOMP.mp.OMP(proj, stop)

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

    # Build channels and decompose them
    estimation = []
    for rays, ue_pos, ap, ii_ue in zip(Rays, UE_pos, AP_selected, range(len(UE_pos))):
        #antenna_UE.uncertainty = orientations_AP[ap]
        antenna_AP.uncertainty = orientations_AP[ap]
        channel.build(rays)
        MM = []
        print("Iteration: {}/{}".format(ii_ue, samples))
        if link == "up":
            for cdbk_AP, Linv in zip(cdbks_AP, LLinv):
                MMM = []
                antenna_AP.set_codebook(cdbk_AP)
                channel.set_corr(np.dot(np.conj(antenna_AP.codebook.T), antenna_AP.codebook))
                for cdbk_UE in cdbks_UE:
                    antenna_UE.set_codebook(cdbk_UE)
                    MMM.append(np.dot(Linv, channel.measure()))
                MM.append(MMM)
        else:
            for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
                MMM = []
                antenna_UE.set_codebook(cdbk_UE)
                channel.set_corr(np.conj(antenna_UE.codebook.T), antenna_UE.codebook)
                for cdbk_AP in cdbks_AP:
                    antenna_AP.set_codebook(cdbk_AP)
                    MM.append(np.dot(Linv, channel.measure()))
                MM.append(MMM)
        M = np.concatenate([np.asarray(MMM).transpose([1, 0, 2]) for MMM in MM], axis=0)
        M_U = np.tensordot(np.tensordot(np.tensordot(M, L_invW_U.conj(), axes=(0, 0)), F_U.conj(), axes=(0, 0)), E_conv_U.conj(), axes=(0, 0))
        tic = time()
        I, alpha = alg(M_U.reshape([-1]))
        toc = time()-tic
        if method == "OMP":
            I = [list(np.unravel_index(ii, [x.shape[1] for x in X])) for ii in I]
        Alpha = []
        Power = []
        DoA = []
        DoD = []
        ToF = []
        for a, iii in zip(alpha, I):
            Alpha.append(a)
            Power.append(20*np.log10(np.linalg.norm(a)))
            ii_component = 0
            if link == "up":
                xoa, yoa = [angles_AP[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
            else:
                xoa, yoa = [angles_UE[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
            zoa = xoa**2 + yoa**2
            if zoa > 1:
                xoa, yoa = xoa/np.sqrt(zoa), yoa/np.sqrt(zoa)
                zoa = 0
            else:
                zoa = np.sqrt(1-zoa)
            doa = np.array([xoa, yoa, zoa])
            DoA.append(doa)
            ii_component += 2
            if link == "up":
                xod, yod = [angles_UE[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
            else:
                xod, yod = [angles_AP[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
            zod = xod**2 + yod**2
            if zod > 1:
                xod, yod = xod/np.sqrt(zod), yod/np.sqrt(zod)
                zod = 0
            else:
                zod = np.sqrt(1-zod)
            dod = np.array([xod, yod, zod])
            DoD.append(dod)
            ii_component += 2
            tof = delays[iii[ii_component]]
            ToF.append(tof)
        Alpha = np.array(Alpha)
        Power = np.array(Power)
        DoA = np.array(DoA)
        DoD = np.array(DoD)
        if link == "up":
            DoA = antenna_AP.uncertainty.apply(DoA)
            DoD = antenna_UE.uncertainty.apply(DoD)
        else:
            DoA = antenna_UE.uncertainty.apply(DoA)
            DoD = antenna_AP.uncertainty.apply(DoD)
        ToF = np.array(ToF)/B
        TDoF = ToF - ToF[0]
        DDoF = TDoF*c
        DoA_az, DoA_el = pywarraychannels.em.cartesian2polar(DoA[0])
        print([np.rad2deg(a) for a in [DoA_az, DoA_el]])
        print(rays.first(5))
        estimation.append({
            "Alpha_r": np.real(Alpha).tolist(), "Alpha_i": np.imag(Alpha).tolist(), "Power": Power.tolist(),
            "DoA": DoA.tolist(), "DoD": DoD.tolist(),
            "DDoF": DDoF.tolist(), "CTime": toc})

    with open(
        "data/{}/AH/paths-dict/single_{}_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
            set, method, dict_style, N_UE, N_AP, N_M_UE, N_M_AP, Q, p_t_dBm, int(10*K_res)),
        "w") as f:
        f.write(json.dumps(estimation))
