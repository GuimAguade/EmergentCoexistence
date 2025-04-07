import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.special

# CODE TO EXPLORE THE CONDITION NUMBER OF THE INTERACTION MATRICES OF COMMUNITIES WITH AND WITHOUT EMERGENT COEXISTENCE AS A FUNCTION OF THEIR DIVERSITY
# GUIM AGUADE-GORGORIÓ APR 2025

##########################################################################################################################

# CONDITION NUMBER OF A MATRIX AS THE RATE BETWEEN LARGEST AND SMALLEST SINGULAR VALUES

def condition_number(A):
    U, S, Vt = np.linalg.svd(A)  # Compute singular values
    sigma_max = np.max(S)
    sigma_min = np.min(S)
    if sigma_min == 0:
        return np.inf  # Ill-conditioned matrix (singular or nearly singular)

    return sigma_max / sigma_min  # Condition number κ(A)
    
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
condition_data = []
x_data_EC = []
condition_data_EC = []
mu_data_EC = []
sigma_data_EC = []
mean_off_diagonal = []
std_off_diagonal = []


for i in range(frac):
    
    print(i," / ", frac)

    # SUBSET SIZE    
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
        mean_off_diagonal.append(np.mean(Aprime))
        std_off_diagonal.append(np.std(Aprime))
        
        # DOES THIS SUBSET LEAD TO A FEASIBLE STATE OF POSITIVE ABUNDANCES?
        INV = np.linalg.inv(-Aprime)
        row_sums = np.sum(INV, axis=1)

        if np.all(row_sums > 0):
            # IF SO, CHECK FOR STABILITY
            J = np.zeros((S, S))
            for ro in range(S):
                for co in range(S):
                    J[ro, co] = row_sums[ro] * Aprime[ro, co]
            
            # ARE ALL EIGENVALUES NEGATIVE (LINEARLY STABLE STATE)              
            if np.max(np.real(np.linalg.eigvals(J))) < 0:
        
                # IS THERE EMERGENT COEXISTENCE IN THIS STATE? (PURPLE DOTS)
                if np.any(Aprime<-1):
                    
                    # RECORD DIVERSITY, STATISTICS, CONDITION NUMBER
                    x_data_EC.append(S)
                    mu_data_EC.append(np.mean(Aprime))
                    sigma_data_EC.append(np.std(Aprime))
                    condition_data_EC.append(condition_number(Aprime))
                
                else: # A STATE WITHOUT EMERGENT COEXISTENCE / NO EXCLUDING ELEMENTS 
                    x_data.append(S)
                    condition_data.append(condition_number(Aprime))    
    
Smath = np.arange(3, Smax)

# PREDICTIONS OF MINIMAL COLLECTIVITY BASED ON SINGULAR VALUES OF A RANDOM MATRIX (ESM II.B.2)

# smallest max singular value
sMmin = np.zeros(len(Smath))

# largest min singular value
smmax = np.zeros(len(Smath))

min_k = np.zeros(len(Smath))



for i in range(len(Smath)):
    Sp = Smath[i]
    mumax = np.max(np.array(mu_data_EC))
    sigmamax = np.max(np.array(sigma_data_EC))    
    mumin = np.min(np.array(mu_data_EC))
    sigmamin = np.min(np.array(sigma_data_EC))
    sMmin[i] = abs(mumin) + (2*sigmamin*np.sqrt(Sp))
    smmax[i] = sigmamax / (np.sqrt(Sp))
    min_k[i] =  sMmin[i]/smmax[i]

plt.figure(figsize=(6, 5))
plt.scatter(x_data, condition_data, color='lightgray', alpha=0.05)
plt.scatter(x_data_EC, condition_data_EC, color='purple', alpha=0.1)
plt.plot(Smath, min_k, linestyle="--", color="firebrick")
plt.axhline(1, color='black', linestyle='--')
plt.xlabel("S", fontsize=12)
plt.ylabel("cond", fontsize=12)
plt.tight_layout()
#plt.ylim(-1.1, 0.1)  # Set minimum y to -0.003, maximum is auto
plt.savefig("Condition_Figure3B.png", format="png")# dpi=100)  # Save as SVG with high DPI
plt.show()


