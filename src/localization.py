import numpy as np
from itertools import permutations

def localization_single_ap(AP_pos, dirs, dist_diff, classi, weights=None):
    """Localize the user using a single ap.

    UE_pos = localization_single_ap(AP_pos, dirs, time_diff, weights = None)

    INPUT:
        AP_pos: 3D position of the access point
        dirs: array of unitary path directions (main path should be LoS)
        time_diff: time difference with the main path time_diff[0] = 0
        classi: list with path classification ("s" for LoS, "v" for wall reflection, "h" for floor/ceiling reflection and "x" for spurious path)
        weights: weights (possitive) to be used for the UE_pos estimation, we
        recommend using the path power as weight

    OUTPUT:
        UE_pos: Estimated 3D position of the user

    """
    # Check if first path is LoS
    if classi[0] != "s":
        return
    # Define weights
    if weights is None:
        weights = np.ones(len(dirs))
    # Dump every variable to a numpy array
    AP_pos, dirs, dist_diff, weights = [np.asarray(a) for a in [AP_pos, dirs, dist_diff, weights]]
    # Compute angle functions relative to the LoS
    cosa_el = np.sqrt(1-np.power(dirs[:, 2], 2))
    cosa_diff_az = np.dot(
        dirs[:, :2]/cosa_el[:, np.newaxis],
        dirs[0, :2]/cosa_el[0])
    sina_el = dirs[:, 2]
    # Estimation equations system
    scalar = 0
    norm2 = 0
    for ii_path in range(1, len(dirs)):
        if classi[ii_path] == "h":
            scalar += weights[ii_path]*(cosa_el[0]-cosa_el[ii_path])*cosa_el[ii_path]*dist_diff[ii_path]
            norm2 += weights[ii_path]*np.power((cosa_el[0]-cosa_el[ii_path]), 2)
        elif classi[ii_path] == "v":
            scalar += weights[ii_path]*(sina_el[0]-sina_el[ii_path])*sina_el[ii_path]*dist_diff[ii_path]
            norm2 += weights[ii_path]*np.power((sina_el[0]-sina_el[ii_path]), 2)
    # Ranging
    if norm2 > 0:
        d = scalar/norm2
        return AP_pos+dirs[0, :]*d
    else:
        return

def localization_single_ap_NLoS(AP_pos, dirs, dirs_D, dist_diff, weights=None):
    """Localize the user using a single ap.

    UE_pos = localization_single_ap(AP_pos, dirs, time_diff, weights = None)

    INPUT:
        AP_pos: 3D position of the access point
        dirs: array of unitary path directions
        time_diff: time difference with any reference
        weights: weights (possitive) to be used for the UE_pos estimation, we
        recommend using the path power as weight

    OUTPUT:
        UE_pos: Estimated 3D position of the user

    """
    if weights is None:
        weights = np.ones(len(dirs))
    # Dump every variable to a numpy array
    AP_pos, dirs, dist_diff, weights = [np.asarray(a) for a in [AP_pos, dirs, dist_diff, weights]]
    # Define weights
    TVT = np.zeros((4, 4))
    TVw = np.zeros(4)
    I = np.eye(3)
    T = np.zeros((3, 4))
    T[:, :3] += I
    for dir_A, dir_D, d_diff, weight in zip(dirs, dirs_D, dist_diff, weights):
        v_AD = dir_A + dir_D
        v_AD_norm2 = np.dot(v_AD, v_AD)
        ImV = I - v_AD[:, np.newaxis]*v_AD[np.newaxis, :]/v_AD_norm2
        T[:, 3] = dir_D
        w = AP_pos - dir_D*d_diff
        TVT += weight*np.dot(np.dot(T.T, ImV), T)
        TVw += weight*np.dot(np.dot(T.T, ImV), w)
    # Solve
    try:
        return np.linalg.solve(TVT, TVw)[:3]
    except:
        return

def localization_eusipco(AP_pos, dirs, dist_diff, classi, weights=None):
    """Localize the user using a single ap.

    UE_pos = localization_single_ap(AP_pos, dirs, time_diff, weights = None)

    INPUT:
        AP_pos: 3D position of the access point
        dirs: array of unitary path directions
        dist_diff: distance difference with any reference
        weights: weights (possitive) to be used for the UE_pos estimation, we
        recommend using the path power as weight

    OUTPUT:
        UE_pos: Estimated 3D position of the user

    """
    if weights is None:
        weights = np.ones(len(dirs))
    # Dump every variable to a numpy array
    AP_pos, dirs, dist_diff, weights = [np.asarray(a) for a in [AP_pos, dirs, dist_diff, weights]]
    #Proj matrix dictionary
    Proj = {
        "s": np.eye(3),
        "v": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
        "h": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        "x": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    }
    # Initialization
    A = np.zeros((4, 4))
    b = np.zeros(4)
    for dir, d_diff, weight, c in zip(dirs, dist_diff, weights, classi):
        E = np.concatenate([np.eye(3), -dir[:, np.newaxis]], axis=1)
        A += weight*np.dot(E.T, np.dot(Proj[c], E))
        b += weight*d_diff*np.dot(dir, np.dot(Proj[c], E))
    try:
        return AP_pos + np.linalg.solve(A, b)[:3]
    except:
        return
