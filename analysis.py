import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import newton
plt.rcParams.update({'font.size': 12.5})

folder = 'res/working/'
ext = '.dat'
show = False
do_computation_PMF = False
save_pics = False
beta = 1.
N = 500
n_F = n_R = N
tot_records = 750
tot_t = 0.75
time = np.linspace(0, tot_t, tot_records)


def g_i(x, delta_A):
    return 1./(1. + np.exp(x - delta_A))

def h_j(x, delta_A):
    return 1./(1. +  np.exp(x + delta_A))

# Implicit relation
def f(x, W_F, W_B):
    sum_g = np.sum([g_i(y, x) for y in W_F])
    sum_h = np.sum([h_j(y, x) for y in W_B])
    return sum_g - sum_h


def Bidirectional_Method(ΔF, W_F_t, W_B_t, W_F, W_B):
    return -np.log(np.mean(np.exp(-W_F_t)/(1.+np.exp(-(W_F-ΔF)))) + np.mean(np.exp(-(W_B_t+ΔF))/(1.+np.exp(-(W_B+ΔF)))))

W_f = np.loadtxt(folder + 'FORWARD_' + f'Work' + ext)
W_b = np.loadtxt(folder + 'BACKWARD_' + f'Work' + ext)

Work = (np.mean(W_f[-1, :]) + np.mean(W_b[-1, :]))/2
print(f'Work = {Work}')

print(f'Var W_f = {np.var(W_f[-1, :])}\nmean = {np.mean(W_f[-1, :])}\nmean(exp) = {np.mean(np.exp(-W_f[-1, :]))}')
print(f'Var W_b = {np.var(W_b[-1, :])}\nmean = {np.mean(W_b[-1, :])},\nmean(exp) = {np.mean(np.exp(-W_b[-1, :]))}')

tot_work_F = W_f[-1, :]
tot_work_B = W_b[-1, :]

ΔF = newton(f, x0=5, args=(tot_work_F, tot_work_B), maxiter=1000)
print(f'At time τ: BAR method outputs ΔF = {ΔF}')

Delta_F_t = np.array([Bidirectional_Method(ΔF, W_f[t, :], W_b[t, :], tot_work_F.mean(), tot_work_B.mean()) for t in range(tot_records)])

if show:
    nf, bins_f = np.histogram(tot_work_F, density=True, bins=30)
    nb, bins_b = np.histogram(tot_work_B, density=True, bins=bins_f)

    plt.scatter((bins_f[1:]+bins_f[:-1])/2, nf, label='F')
    plt.scatter((bins_b[1:]+bins_b[:-1])/2, nb*np.exp(-((bins_b[1:]+bins_b[:-1])/2+6.523619906750094)), label='B')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.title(r'$Work:\: \overline{W}$')
    plt.plot(W_f.mean(axis=1), label='F')
    plt.plot(W_b.mean(axis=1), label='B')
    plt.show()

    plt.title(r'$\Delta F_t$')
    plt.plot(Delta_F_t)
    plt.show()

Jarz_eq_f = -np.log(np.mean(np.exp(-W_f), axis=1))
Jarz_eq_b = -np.log(np.mean(np.exp(-W_b), axis=1)) + ΔF                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   


time_steps = 10
plt.figure(figsize=(10, 4))
plt.scatter(time[::time_steps], Delta_F_t[::time_steps], color='tab:blue', 
            marker=10, label='Optimized')
plt.scatter(time[::time_steps], Jarz_eq_f[::time_steps], color='tab:green', 
            marker=9, label='Forward Jarzynski')
plt.scatter(time[::time_steps], Jarz_eq_b[::time_steps], color='tab:red', 
            marker=8, label='Backward Jarzynski')
plt.legend()
plt.xlabel('Time')
plt.ylabel(r'$\Delta F_t\: (k_B T)$')
plt.tight_layout()
if save_pics:
    plt.savefig('imgs/Figure1.png', dpi=500, transparent=True)
plt.show()



###################
### Compute PMF ###
###################

def H0(z):
    return 5.*(z**4.) - 10.*(z**2) + 3.*z

def V_t(z, z_t):
    return (15/2.) * (z-z_t)**2

def Hamiltonian(z, z_t):
    return H0(z) + V_t(z, z_t)


z_t_forw = np.linspace(-1.5, 1.5, tot_records)#.tolist() + [1.5]
z_t_back = np.linspace(1.5, -1.5, tot_records)#.tolist() + [-1.5]

rho0_f = np.zeros(z_t_forw.shape)
rho0_b = np.zeros(z_t_back.shape)

data_f = np.loadtxt(folder + 'FORWARD_data.dat')[1:, :]
data_b = np.loadtxt(folder + 'BACKWARD_data.dat')[1:, :]

average_W_f = np.mean(W_f, axis=1)
average_W_b = np.mean(W_b, axis=1)

lambda_min = np.min(z_t_forw)
lambda_max = np.max(z_t_forw)
num_bins = 110
lambda_bins = np.linspace(lambda_min, lambda_max, num_bins)

pmf = np.zeros(num_bins)
pmf_b = np.zeros(num_bins)
pmf_good = np.zeros(num_bins)
delta = (lambda_max - lambda_min) / num_bins

if not os.path.exists(folder + 'PMF_F.dat') or do_computation_PMF:

    for i in range(num_bins):
        lambda_i = lambda_bins[i]
        # lambda_i_b = lambda_bins[num_bins-1-i]
        num, num_b, num_good_F, num_good_B = 0, 0, 0, 0
        den, den_b, den_good = 0, 0, 0
        for tt in range(tot_records):
            nf, bins_f = np.histogram(data_f[tt, :], density=True)
            real_bins_f = (bins_f[:-1]+bins_f[1:])/2
            mask_f = np.logical_and(real_bins_f >= lambda_i - delta/2, real_bins_f < lambda_i + delta/2)
            distrib_f = nf[mask_f]
            
            if sum(mask_f) > 0:
                num += distrib_f*np.exp(-average_W_f[tt]) * np.exp(Jarz_eq_f[tt])
                num_good_F += (distrib_f*np.exp(-average_W_f[tt])/(1.+np.exp(ΔF-average_W_f[-1])))*np.exp(Delta_F_t[tt])
            den += np.exp(Jarz_eq_f[tt] - V_t(lambda_i, z_t_forw[tt]))
                
            nb, bins_b = np.histogram(data_b[tt, :], density=True)
            real_bins_b = (bins_b[:-1]+bins_b[1:])/2
            mask_b = np.logical_and(real_bins_b >= lambda_i - delta/2, real_bins_b < lambda_i + delta/2)
            distrib_b = nb[mask_b]

            if sum(mask_b) > 0:
                num_b += distrib_b*np.exp(-average_W_b[tt]-average_W_b[-1]) * np.exp(Jarz_eq_b[tt])
                num_good_B += (distrib_b*np.exp(-average_W_b[tt]+average_W_b[-1])/(1.+np.exp((ΔF+average_W_b[-1]))))*np.exp(Delta_F_t[tt])
            den_b += np.exp(Jarz_eq_b[tt] - V_t(lambda_i, z_t_back[tt]))

            den_good += np.exp(Delta_F_t[tt] - V_t(lambda_i, z_t_forw[tt]))
            
        pmf[i] = -np.log(num/den)
        pmf_b[i] = -np.log(num_b/den_b)
        pmf_good[i] = -np.log((num_good_F + num_good_B)/den_good)

    PMF_minimum = min(pmf.min(), pmf_good.min())

    np.savetxt(folder + 'PMF_F' + ext, pmf-PMF_minimum)
    np.savetxt(folder + 'PMF_B' + ext, pmf_b-PMF_minimum)
    np.savetxt(folder + 'PMF_MA' + ext, pmf_good-PMF_minimum)
    np.savetxt(folder + 'lambda_bins' + ext, lambda_bins)

else:
    pmf = np.loadtxt(folder + 'PMF_F' + ext)
    pmf_b = np.loadtxt(folder + 'PMF_B' + ext)
    pmf_good = np.loadtxt(folder + 'PMF_MA' + ext)
    lambda_bins = np.loadtxt(folder + 'lambda_bins' + ext)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ts = 2
ax[0].scatter(lambda_bins[::ts], (pmf)[::ts], color='tab:green', 
            marker=9, label='Forward: Unidirectional')
ax[0].scatter(lambda_bins[::-ts], (pmf_b)[::ts], color='tab:red', 
            marker=8, label='Backward: Unidirectional')
ax[1].scatter(lambda_bins[::ts], (pmf_good)[::ts], color='tab:blue', 
            marker=10, label='Optimized: Bidirectional')
ax[1].plot(lambda_bins[np.logical_and(lambda_bins > -1, lambda_bins <= 1.01)][::ts], 
           V_t(lambda_bins, 0)[np.logical_and(lambda_bins > -1, lambda_bins <= 1.01)][::ts]+pmf_good[np.where(lambda_bins>0)[0][0]], 
           ls='--', color='orange')
ax[0].set_ylabel(r'$PMF\: (k_B T)$')
y_lims = ax[0].get_ylim()
ax[1].set_yticks([])
ax[1].set_ylim(*y_lims)
ax[0].legend()
ax[1].legend(loc='upper left')
# plt.tight_layout()
plt.subplots_adjust(wspace=0.02, bottom=0.15)
fig.text(0.5+0.01, 0.04, 'Position', ha='center', va='center', transform=fig.transFigure)
if save_pics:
    plt.savefig('imgs/Figure2.png', dpi=500, transparent=True)
plt.show()
