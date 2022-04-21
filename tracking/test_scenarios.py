import numpy as np
import matplotlib.pyplot as plt
from tracking import import_data
#from orbital_module import weibeOrbit2radians, orbitl2velocity, AaRdR_to_v

#%% Inputs
file = "nfft_50k"
fig_path = "matlab_code_figs"
sat = 1

#%%
file_truth = []
file_false = file+"/false.txt"
file_entire_orb = []
for i in range(1,6):
    file_truth.append(file+"/truth"+f"{i}.txt")
    file_entire_orb.append(file+"/entireOrbit"+f"{i}.txt")

t_true = []; R_true = []; A_true = []; E_true = []; dR_true = []; SNR_true = []

for i,file_ in enumerate(file_truth):
    data = import_data(file_)
    t_true.append(data[0])
    R_true.append(data[1])
    A_true.append(data[2])
    E_true.append(data[3])
    dR_true.append(data[4])
    SNR_true.append(data[5])
    
t_false,R_false,A_false,E_false,dR_false,SNR_false = import_data(file_false)

t_orb,R_orb,A_orb,E_orb,dR_orb,_ = import_data(file_entire_orb[sat])


#%% Test 1
colors = ["black","blue","orange","cyan","red","magenta"]
fontsiz = 14
fig, axs = plt.subplots(2,2, sharex=True,sharey=False,figsize=(14,10))
fig.subplots_adjust(left=0.1, wspace=0.3)
#fig.suptitle(f'[$\\beta$, $e$, $\Omega$, $i$, $\omega$] = {weibel_orbit_elem}',fontsize=29)

axs[0][0].plot(t_orb,R_orb,color=colors[sat+1], label=f"Satellite {sat+1}")
#axs[0][0].scatter(t_true[0],R_true[0], marker='.',color=colors[1],label=f"Satellite 1")
axs[0][0].set_ylabel("Range [km]",fontsize=fontsiz)
axs[0][0].grid(True)
#axs[0][0].yaxis.set_label_coords(x_, y_)
axs[0][0].set_ylim([np.min(R_true[sat])-2,np.max(R_true[sat])+2])

axs[1][0].plot(t_orb,A_orb,color=colors[sat+1], label=f"Satellite {sat+1}")
#axs[1][0].scatter(t_true[0],A_true[0], marker='.',color=colors[1],label=f"Satellite 1")
axs[1][0].set_ylabel("Azimuth [degrees]",fontsize=fontsiz)
axs[1][0].set_xlabel("Time [s]",fontsize=fontsiz)
axs[1][0].set_xlim([np.min(t_true[sat])-2,np.max(t_true[sat])+2])
axs[1][0].grid(True)
#axs[1][0].yaxis.set_label_coords(x_, y_)
axs[1][0].set_ylim([np.min(A_true[sat])-2,np.max(A_true[sat])+2])

axs[0][1].plot(t_orb,E_orb,color=colors[sat+1], label=f"Satellite {sat+1}")
#axs[0][1].scatter(t_true[0],E_true[0], marker='.',color=colors[1],label=f"Satellite 1")
axs[0][1].set_ylabel("Elevation [degrees]",fontsize=fontsiz)
axs[0][1].grid(True)
#axs[0][1].yaxis.set_label_coords(x_, y_)
axs[0][1].set_ylim([np.min(E_true[sat])-2,np.max(E_true[sat])+2])

axs[1][1].plot(t_orb,dR_orb,color=colors[sat+1], label=f"Satellite {sat+1}")
#axs[1][1].scatter(t_true[0],dR_true[0], marker='.',color=colors[1],label=f"Satellite 1")
axs[1][1].set_ylabel("Radial Velocity [km/s]",fontsize=fontsiz)
axs[1][1].set_xlabel("Time [s]",fontsize=fontsiz)
axs[1][1].set_xlim([np.min(t_true[sat])-2,np.max(t_true[sat])+2])
axs[1][1].set_ylim([np.min(dR_true[sat])-2,np.max(dR_true[sat])+2])
axs[1][1].grid(True)
#axs[1][1].yaxis.set_label_coords(x_, y_)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0,fontsize=fontsiz)
plt.tight_layout()
#plt.savefig(fig_path+f"/entire_orb_zoom_{file}.pdf")
plt.show()


#%% Test 2

fig, axs = plt.subplots(2,2, sharex=True,sharey=False,figsize=(14,10))
fig.subplots_adjust(left=0.1, wspace=0.3)
#fig.suptitle(f'[$\\beta$, $e$, $\Omega$, $i$, $\omega$] = {weibel_orbit_elem}',fontsize=29)


#axs[0][0].scatter(t_false,R_false, marker='.',color=colors[0],label=f"False alarm")
axs[0][0].set_ylabel("Range [km]",fontsize=fontsiz)
axs[0][0].grid(True)
#axs[0][0].yaxis.set_label_coords(x_, y_)
#axs[0][0].set_ylim([np.min(R_true[0])-2,np.max(R_true[0])+2])

#axs[1][0].scatter(t_false,A_false, marker='.',color=colors[0],label=f"False alarm")
axs[1][0].set_ylabel("Azimuth [degrees]",fontsize=fontsiz)
axs[1][0].set_xlabel("Time [s]",fontsize=fontsiz)
#axs[1][0].set_xlim([np.min(t_true[0])-2,np.max(t_true[0])+2])
axs[1][0].grid(True)
#axs[1][0].yaxis.set_label_coords(x_, y_)
#axs[1][0].set_ylim([np.min(A_true[0])-2,np.max(A_true[0])+2])

#axs[0][1].scatter(t_false,E_false, marker='.',color=colors[0],label=f"False alarm")
axs[0][1].set_ylabel("Elevation [degrees]",fontsize=fontsiz)
axs[0][1].grid(True)
#axs[0][1].yaxis.set_label_coords(x_, y_)
#axs[0][1].set_ylim([np.min(E_true[0])-2,np.max(E_true[0])+2])

#axs[1][1].scatter(t_false,dR_false, marker='.',color=colors[0],label=f"False alarm")
axs[1][1].set_ylabel("Radial Velocity [km/s]",fontsize=fontsiz)
axs[1][1].set_xlabel("Time [s]",fontsize=fontsiz)
#axs[1][1].set_xlim([np.min(t_true[0])-2,np.max(t_true[0])+2])
#axs[1][1].set_ylim([np.min(dR_true[0])-2,np.max(dR_true[0])+2])
axs[1][1].grid(True)
#axs[1][1].yaxis.set_label_coords(x_, y_)

for i in range(2):
    axs[0][0].scatter(t_true[i],R_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")
    axs[1][0].scatter(t_true[i],A_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")
    axs[0][1].scatter(t_true[i],E_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")
    axs[1][1].scatter(t_true[i],dR_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0,fontsize=14)
plt.tight_layout()
#plt.savefig(fig_path+f"/sat1_sat2_{file}.pdf")
plt.show()

#%% Test 3

fig, axs = plt.subplots(2,2, sharex=True,sharey=False,figsize=(14,10))
fig.subplots_adjust(left=0.1, wspace=0.3)
#fig.suptitle(f'[$\\beta$, $e$, $\Omega$, $i$, $\omega$] = {weibel_orbit_elem}',fontsize=29)


axs[0][0].scatter(t_false,R_false, marker='.',color=colors[0],label=f"False alarm")
axs[0][0].set_ylabel("Range [km]",fontsize=fontsiz)
axs[0][0].grid(True)
#axs[0][0].yaxis.set_label_coords(x_, y_)
#axs[0][0].set_ylim([np.min(R_true[0])-2,np.max(R_true[0])+2])

axs[1][0].scatter(t_false,A_false, marker='.',color=colors[0],label=f"False alarm")
axs[1][0].set_ylabel("Azimuth [degrees]",fontsize=fontsiz)
axs[1][0].set_xlabel("Time [s]",fontsize=fontsiz)
#axs[1][0].set_xlim([np.min(t_true[0])-2,np.max(t_true[0])+2])
axs[1][0].grid(True)
#axs[1][0].yaxis.set_label_coords(x_, y_)
#axs[1][0].set_ylim([np.min(A_true[0])-2,np.max(A_true[0])+2])

axs[0][1].scatter(t_false,E_false, marker='.',color=colors[0],label=f"False alarm")
axs[0][1].set_ylabel("Elevation [degrees]",fontsize=fontsiz)
axs[0][1].grid(True)
#axs[0][1].yaxis.set_label_coords(x_, y_)
#axs[0][1].set_ylim([np.min(E_true[0])-2,np.max(E_true[0])+2])

axs[1][1].scatter(t_false,dR_false, marker='.',color=colors[0],label=f"False alarm")
axs[1][1].set_ylabel("Radial Velocity [km/s]",fontsize=fontsiz)
axs[1][1].set_xlabel("Time [s]",fontsize=fontsiz)
#axs[1][1].set_xlim([np.min(t_true[0])-2,np.max(t_true[0])+2])
#axs[1][1].set_ylim([np.min(dR_true[0])-2,np.max(dR_true[0])+2])
axs[1][1].grid(True)
#axs[1][1].yaxis.set_label_coords(x_, y_)

for i in range(2):
    axs[0][0].scatter(t_true[i],R_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")
    axs[1][0].scatter(t_true[i],A_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")
    axs[0][1].scatter(t_true[i],E_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")
    axs[1][1].scatter(t_true[i],dR_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0,fontsize=14)
plt.tight_layout()
plt.savefig(fig_path+f"/sat1_sat2_false_{file}.pdf")
plt.show()


#%% Test 4
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

fig, axs = plt.subplots(2,2, sharex=True,sharey=False,figsize=(14,10))
fig.subplots_adjust(left=0.1, wspace=0.3)
#fig.suptitle(f'[$\\beta$, $e$, $\Omega$, $i$, $\omega$] = {weibel_orbit_elem}',fontsize=29)


#axs[0][0].scatter(t_false,R_false, marker='.',color=colors[0],label=f"False alarm")
axs[0][0].set_ylabel("Range [km]",fontsize=fontsiz)
axs[0][0].grid(True)
#axs[0][0].yaxis.set_label_coords(x_, y_)
#axs[0][0].set_ylim([np.min(R_true[0])-2,np.max(R_true[0])+2])

#axs[1][0].scatter(t_false,A_false, marker='.',color=colors[0],label=f"False alarm")
axs[1][0].set_ylabel("Azimuth [degrees]",fontsize=fontsiz)
axs[1][0].set_xlabel("Time [s]",fontsize=fontsiz)
#axs[1][0].set_xlim([np.min(t_true[0])-2,np.max(t_true[0])+2])
axs[1][0].grid(True)
#axs[1][0].yaxis.set_label_coords(x_, y_)
#axs[1][0].set_ylim([np.min(A_true[0])-2,np.max(A_true[0])+2])

#axs[0][1].scatter(t_false,E_false, marker='.',color=colors[0],label=f"False alarm")
axs[0][1].set_ylabel("Elevation [degrees]",fontsize=fontsiz)
axs[0][1].grid(True)
#axs[0][1].yaxis.set_label_coords(x_, y_)
#axs[0][1].set_ylim([np.min(E_true[0])-2,np.max(E_true[0])+2])

#axs[1][1].scatter(t_false,dR_false, marker='.',color=colors[0],label=f"False alarm")
axs[1][1].set_ylabel("Radial Velocity [km/s]",fontsize=fontsiz)
axs[1][1].set_xlabel("Time [s]",fontsize=fontsiz)
#axs[1][1].set_xlim([np.min(t_true[0])-2,np.max(t_true[0])+2])
#axs[1][1].set_ylim([np.min(dR_true[0])-2,np.max(dR_true[0])+2])
axs[1][1].grid(True)
#axs[1][1].yaxis.set_label_coords(x_, y_)

for i in range(5):
    axs[0][0].plot(t_orb[i],R_orb[i],color=colors[i+1],label=f"Satellite {i+1}")
    axs[1][0].plot(t_orb[i],A_orb[i],color=colors[i+1],label=f"Satellite {i+1}")
    axs[0][1].plot(t_orb[i],E_orb[i],color=colors[i+1],label=f"Satellite {i+1}")
    axs[1][1].plot(t_orb[i],dR_orb[i],color=colors[i+1],label=f"Satellite {i+1}")

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0,fontsize=14)
plt.tight_layout()
plt.savefig(fig_path+f"/multiple_orbits.pdf")
plt.show()

#%% Plots 4
'''
fig, axs = plt.subplots(2,2, sharex=True,sharey=False,figsize=(14,10))
fig.subplots_adjust(left=0.1, wspace=0.3)
#fig.suptitle(f'[$\\beta$, $e$, $\Omega$, $i$, $\omega$] = {weibel_orbit_elem}',fontsize=29)
x_ = -0.15; y_ = 0.5

axs[0][0].plot(t_orb,R_orb, label="Exact orbit")
axs[0][0].scatter(t_true[0],R_true[0], marker='.',color=colors[1],label=f"Satellite 1")
axs[0][0].set_ylabel("Range [km]")
axs[0][0].grid(True)

axs[1][0].plot(t_orb,A_orb, label="Exact orbit")
axs[1][0].scatter(t_true[0],A_true[0], marker='.',color=colors[1],label=f"Satellite 1")
axs[1][0].set_ylabel("Azimuth [degrees]")
axs[1][0].set_xlabel("Time [s]")
axs[1][0].grid(True)

axs[0][1].plot(t_orb,E_orb, label="Exact orbit")
axs[0][1].scatter(t_true[0],E_true[0], marker='.',color=colors[1],label=f"Satellite 1")
axs[0][1].set_ylabel("Elevation [degrees]")
axs[0][1].grid(True)

axs[1][1].plot(t_orb,dR_orb, label="Exact orbit")
axs[1][1].scatter(t_true[0],dR_true[0], marker='.',color=colors[1],label=f"Satellite 1")
axs[1][1].set_ylabel("Radial Velocity [km/s]")
axs[1][1].set_xlabel("Time [s]")
axs[1][1].grid(True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0,fontsize=14)
plt.tight_layout()
plt.savefig(fig_path+f"/entire_orb_{file}.pdf")
plt.show()

#%% Plots 2 (SNR)
plt.scatter(t_false,SNR_false, marker='.',color=colors[0], label="False alarm")
for i in range(5):
    plt.scatter(t_true[i],SNR_true[i], marker='.',color=colors[i+1],label=f"Satellite {i+1}")
plt.grid(True)
plt.ylabel('SNR')
plt.xlabel('Time [s]')
plt.xlim([0,80])
plt.legend(loc="upper right")
plt.savefig(fig_path+f'/{file}.pdf')
plt.show()
'''