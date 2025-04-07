import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.special

# CODE TO MEASURE THE CORRELATION BETWEEN DIRECT AND NET EFFECTS IN COMMUNITIES WITH AND WITHOUT EMERGENT COEXISTENCE
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

# DEFINE DATA TO FILL 

x_data = []
corr_data = []

x_data_EC = []
corr_data_EC = []


for i in range(frac):
    
    print(i," / ", frac)
    
    S = Smin + int(i*(Smax-Smin)/frac)
    
    corr = 0
    corr_EC = 0
    
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
        if np.all(row_sums > 0):
        
            J = np.zeros((S, S))
            for ro in range(S):
                for co in range(S):
                    J[ro, co] = row_sums[ro] * Aprime[ro, co]
            
            # ARE ALL EIGENVALUES NEGATIVE (LINEARLY STABLE STATE)              
            if np.max(np.real(np.linalg.eigvals(J))) < 0:
        
                pears,_ = pearsonr(-Aprime.flatten(), INV.flatten()) 
    
                if np.any(Aprime>1):
                    corr_data_EC.append(pears)
                    x_data_EC.append(S)
                    
                else: 
                    corr_data.append(pears)
                    x_data.append(S)

     
            
plt.figure(figsize=(6, 5))
plt.scatter(np.array(x_data), np.array(corr_data), color='lightgray', alpha=0.05)
plt.scatter(np.array(x_data_EC), np.array(corr_data_EC), color='purple', alpha=0.1)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("S", fontsize=12)
plt.ylabel("corr", fontsize=12)
plt.tight_layout()
plt.savefig("S_corr.png", format="png")# dpi=100)  # Save as SVG with high DPI
plt.show()


