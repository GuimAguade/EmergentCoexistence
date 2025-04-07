import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# CODE TO MEASURE THE SINGULAR VALUES AND CONDITION NUMBERS IN COMMUNITIES WITH EMERGENT COEXISTENCE
# GUIM AGUADE-GORGORIÓ APR 2025

#####################################################################################################################

# FOLLOWING ESM SECTION I.E.2, WE ARE GOING TO SAMPLE SUBSETS OF DIFFERENT SIZE AND TEST THEIR FEASIBILITY AND STABILITY   

# DIVERSITY RANGE
Smin = 3
Smax = 13
frac = Smax - Smin

# MU RANGE OF FIGURE 2A
mumin = -2.0
mumax = 0.5

# SIGMA RANGE
sigmamin = 0.0
sigmamax = 1.0

# NUMBER OF SAMPLED STATES THAT WILL BE TESTED FOR COEXISTENCE, STABILITY, EMERGENT COEXISTENCE AND COLLECTIVITY
reps = 1000000

x_data = []
smax = []
smin = []


# OBTAIN THE MAX AND MIN SINGULAR VALUES
def condition_number(A):
    U, S, Vt = np.linalg.svd(A)  # Compute singular values
    sigma_max = np.max(S)
    sigma_min = np.min(S)
    if sigma_min == 0:
        return np.inf  # Ill-conditioned matrix (singular or nearly singular)

    return sigma_max, sigma_min  # Condition number κ(A)

# MEASURE MATRIX STATISTICS 
mean_off_diagonal = []
std_off_diagonal = []

for i in range(frac):
    
    print(i," / ", frac)
    
    S = Smin + int(i*(Smax-Smin)/frac)
    
    for j in range(reps):
    
        # GENERATE RANDOM MU,SIGMA PAIR
        mu = np.random.uniform(mumin, mumax)
        sigma = np.random.uniform(sigmamin, sigmamax)
        
        # DEFINE A
        A = np.random.normal(mu, sigma, size=(80, 80))
        
        # INCORPORATE RELATIVE SELF-REGULATION
        np.fill_diagonal(A, -np.ones(80))
                
        # SELECT A RANDOM SUBSET OF SIZE "S^*" WHICH HERE IS SIMPLY "S"
        species = np.random.choice(80, size=S, replace=False)
        
        # MEASURE INTERACTION PROPERTIES OF THIS SUBSET
        Aprime = A[np.ix_(species, species)]
        off_diagonal_elements = Aprime[~np.eye(Aprime.shape[0], dtype=bool)]
        
        # DOES THIS SUBSET LEAD TO A FEASIBLE STATE OF POSITIVE ABUNDANCES?
        INV = np.linalg.inv(-Aprime)
        row_sums = np.sum(INV, axis=1)
        
        # IF SO, CHECK FOR STABILITY
        if np.all(row_sums > 0) and np.any(Aprime <-1):
        
            J = np.zeros((S, S))
            for ro in range(S):
                for co in range(S):
                    J[ro, co] = row_sums[ro] * Aprime[ro, co]
            
            # ARE ALL EIGENVALUES NEGATIVE (LINEARLY STABLE STATE)              
            if np.max(np.real(np.linalg.eigvals(J))) < 0 :
                 
                # OBTAIN SINGULAR VALUES
                sigmaM,sigmam = condition_number(Aprime)
                
                smax.append(sigmaM)
                smin.append(sigmam)
                
                # RECORD DIVERSITY
                x_data.append(S)
                
                mean_off_diagonal.append(np.mean(Aprime))
                std_off_diagonal.append(np.std(Aprime))
                


# GENERATE THE ANALYTICAL PREDICTIONS
                    

Smath = np.arange(3, Smax)

sMmax = np.zeros(len(Smath))
sMmin = np.zeros(len(Smath))

sMmax2 = np.zeros(len(Smath))
sMmin2 = np.zeros(len(Smath))

smmax = np.zeros(len(Smath))
smmin = np.zeros(len(Smath))

smmax2 = np.zeros(len(Smath))
smmin2 = np.zeros(len(Smath))


max_k = np.zeros(len(Smath))
min_k = np.zeros(len(Smath))
min_k2 = np.zeros(len(Smath))

for i in range(len(Smath)):
    
    Sp = Smath[i]
    target_S = Sp  # or any specific value you're searching for
    
    matching_indices = [i for i, x in enumerate(x_data) if x == target_S]
    
    matching_means = [mean_off_diagonal[i] for i in matching_indices]
    
    matching_sigmas = [std_off_diagonal[i] for i in matching_indices]
    
    if matching_means:
        mumax = max(matching_means)
        sigmamax = max(matching_sigmas)
    
        mumin = min(matching_means)
        sigmamin = min(matching_sigmas)
        
    else:
        print(f"No entries found for S = {target_S}")
    
    sMmax2[i] =  (2*sigmamax*np.sqrt(Sp))  + abs(mumax) 
    sMmin2[i] =  (2*sigmamin*np.sqrt(Sp))  + abs(mumin)
    
    smmax[i] = sigmamax / np.sqrt(Sp) 
    
    min_k[i] =  sMmin2[i]/smmax[i]
    

    
                    
plt.figure(figsize=(6, 5))
plt.scatter(x_data, smax, color='green', alpha=0.5)
plt.scatter(x_data, smin, color='firebrick', alpha=0.1)
plt.plot(Smath, sMmax2, color='green', alpha=1)
plt.plot(Smath, sMmin2, color='green', alpha=1)
plt.plot(Smath, smmax, color='firebrick', alpha=1)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("S", fontsize=12)
plt.ylabel("singular values", fontsize=12)
plt.tight_layout()
plt.savefig("S_singular_values.png", format="png")# dpi=100)  # Save as SVG with high DPI
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(np.array(x_data), np.array(smax)/np.array(smin), color='purple', alpha=0.5)
plt.plot(Smath, min_k, color='firebrick', alpha=1)
plt.axhline(1, color='black', linestyle='--')
plt.xlabel("S", fontsize=12)
plt.ylabel("corr", fontsize=12)
plt.tight_layout()
plt.savefig("S_cond_nozoom.png", format="png")# dpi=100)  # Save as SVG with high DPI
plt.show()


