from tracking import import_data
import numpy as np

file = "nfft_15k"
file_truth = []
file_entire_orb = []
for i in range(1,6):
    file_truth.append(file+"/truth"+f"{i}.txt")
    file_entire_orb.append(file+"/entireOrbit"+f"{i}.txt")

t_true = []

for i,file_ in enumerate(file_truth):
    data = import_data(file_)
    t_true.append(data[0])

t_orb = []; R_orb = []; A_orb = []; E_orb = []; dR_orb = []
for i,file_ in enumerate(file_entire_orb):
    data = import_data(file_)
    t_orb.append(data[0])
    R_orb.append(data[1])
    A_orb.append(data[2])
    E_orb.append(data[3])
    dR_orb.append(data[4])
    
    lower = np.where(t_orb[i]==round(t_true[i][0]))[0][0]
    upper = np.where(t_orb[i]==round(t_true[i][-1]))[0][0]
    
    t_orb[i] = t_orb[i][lower:upper+1]
    R_orb[i] = R_orb[i][lower:upper+1]
    A_orb[i] = A_orb[i][lower:upper+1]
    E_orb[i] = E_orb[i][lower:upper+1]
    dR_orb[i] = dR_orb[i][lower:upper+1]

data4 = []
for i in range(len(t_orb)):
    data4.append(np.vstack((t_orb[i],R_orb[i],A_orb[i],E_orb[i],dR_orb[i])).T)
data4 = np.concatenate(data4)

data4 = data4[data4[:, 0].argsort()]