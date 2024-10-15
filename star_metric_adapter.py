"""
Created on Thu Mar 21 07:50:04 2024

@author: XinzeLee
@github: https://github.com/XinzeLee/PANN
         https://github.com/XinzeLee/STAR_Metric_Adapter

@reference:
    1: STAR: One-Stop Optimization for Dual-Active-Bridge Converter With Robustness to Operational Diversity
        Authors: Fanfan Lin, Xinze Li (corresponding author), Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/JESTPE.2024.3392684
    2: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Huai Wang, Xin Zhang, Hao Ma, Changyun Wen and Frede Blaabjerg
        Paper DOI: 10.1109/TIE.2024.3352119

"""

import numpy as np


def current_stress(iL, metric="ipp"):
    assert metric in ["ipp", "irms"]
    if metric == "ipp":
        return max(iL)-min(iL)
    elif metric == "irms":
        return np.sqrt((iL**2).mean())
    

def cu_loss(iL, RL):
    iLrms = current_stress(iL, "irms")
    return iLrms**2*RL
    
    
def locate(v, V):
    """
        Locate the Indices of Switching Moments for DAB Converters Based on vp or vs
    """
    idx = [None]*4
    v0 = v[0]
    for i in range(1, len(v)+1):
        i = i % len(v)
        dV = v[i]-v0
        if (dV > V/2):
            if v0 < -V/2: idx[0] = i
            else: idx[1] = i
            if (dV > V*1.5):
                idx[1] = i
        elif (dV < -V/2):
            if v0 > V/2: idx[2] = i
            else: idx[3] = i
            if (dV < -V*1.5):
                idx[3] = i
        v0 = v[i]
    return idx


def soft_switching(iL, vp, vs, Vin, Vo, threshold=0.):
    # Evaluate soft switching performances of conventional DAB converters
    indices = locate(vp, Vin)
    indices2 = locate(vs, Vo)
    ZVS = np.zeros(8, )
    ZCS = np.zeros(8, )
    
    ##### ZVS of Primary Side Switches
    ZVS[0] = (iL[indices[1]]<=threshold).astype(int) # ZVS: S1
    ZVS[1] = (iL[indices[3]]>=-threshold).astype(int) # ZVS: S2
    ZVS[2] = (iL[indices[2]]>=-threshold).astype(int) # ZVS: S3
    ZVS[3] = (iL[indices[0]]<=threshold).astype(int) # ZVS: S4
    
    ##### ZVS of Secondary Side Switches
    ZVS[4] = (iL[indices2[1]]>=-threshold).astype(int) # ZVS: S5
    ZVS[5] = (iL[indices2[3]]<=threshold).astype(int) # ZVS: S6
    ZVS[6] = (iL[indices2[2]]<=threshold).astype(int) # ZVS: S7
    ZVS[7] = (iL[indices2[0]]>=-threshold).astype(int) # ZVS: S8
    return ZVS, indices, indices2


def power(vp, vs, iL, metric="PL", n=None):
    assert metric in ["PL", "Q"]
    if metric == "PL":
        return (vp*iL).mean()
    elif metric == "Q":
        assert n is not None
        vL = vp-n*vs
        return np.sqrt((vL**2).mean())*np.sqrt((iL**2).mean())
    