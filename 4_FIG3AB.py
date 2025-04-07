import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.special

# CODE TO EXPLORE THE SPECTRAL RADIUS OF THE INTERACTION MATRICES OF COMMUNITIES WITH AND WITHOUT EMERGENT COEXISTENCE AS A FUNCTION OF THEIR DIVERSITY
# GUIM AGUADE-GORGORIÓ APR 2025

##########################################################################################################################

# THE SPECTRAL RADIUS OF THE MATRIX CONSIDERS ONLY THE OFF-DIAGONAL ELEMENTS, AS (I-A)^{-1}= 1 + A + A^2 + ...

def spectral_radius(matrix):
    B = matrix + np.identity(len(matrix)) # BY ADDING AN IDENTITY, WE ARE ERASING THE -1 ELEMENTS OF THE DIAGONAL, SO THAT WE TRANSFORM -I+A TO A.
    # Calculate the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(B)
    # Find the maximum absolute value among the eigenvalues
    return max(abs(eigenvalue) for eigenvalue in eigenvalues)

# TO PLOT FIGURE 3B - A CORRELATION FOR A GIVEN Aprime matrix

def plot_off_diagonal(Aprime, INV):
    # Ensure the matrices are numpy arrays
    Aprime = np.array(Aprime)
    INV = np.array(INV)
    
    # Get the indices of the off-diagonal elements
    off_diag_indices = np.where(~np.eye(Aprime.shape[0], dtype=bool))
    
    # Extract off-diagonal elements
    Aprime_off_diag = Aprime[off_diag_indices]
    INV_off_diag = INV[off_diag_indices]
    
    # Plot the off-diagonal elements
    plt.figure(figsize=(6, 6))
    plt.scatter(Aprime_off_diag, INV_off_diag, color='purple', alpha=0.7, edgecolors='black')
    
    # Labels and title
    plt.xlabel('Aprime_ij')
    plt.ylabel('INV_ij')
    plt.title('Off-Diagonal Elements of Aprime vs INV')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()



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
phi_data = []
x_data_EC = []
phi_data_EC = []
mu_data_EC = []
sigma_data_EC = []
mean_off_diagonal = []
std_off_diagonal = []


for i in range(frac):
    
    print(i," / ", frac)
    
    # SUBSET SIZE
    S = Smin + int(i*(Smax-Smin)/frac)
    
    # SET COUNTERS TO ZERO
    phi = 0
    phi_EC = 0
    entra = 0
    entraEC = 0
    
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
        
        # IF SO, CHECK FOR STABILITY
        if np.all(row_sums > 0):
        
            J = np.zeros((S, S))
            for ro in range(S):
                for co in range(S):
                    J[ro, co] = row_sums[ro] * Aprime[ro, co]
            
            # ARE ALL EIGENVALUES NEGATIVE (LINEARLY STABLE STATE)              
            if np.max(np.real(np.linalg.eigvals(J))) < 0:
        
                # IS THERE EMERGENT COEXISTENCE IN THIS STATE? (PURPLE DOTS)
                if np.any(Aprime<-1):
                    
                    # RECORD DIVERSITY, STATISTICS, SPECTRAL RADIUS
                    x_data_EC.append(S)
                    mu_data_EC.append(np.mean(Aprime))
                    sigma_data_EC.append(np.std(Aprime))
                    phi_data_EC.append(spectral_radius(Aprime))
                    
                    # CODE TO GENERATE FIGURE 3B - CAN BE PERFORMED ONLY ONCE AND THEN COMMENTED FOR THE FULL SIMULATION
                    #if spectral_radius(Aprime) > 2 and S> 6:
                    #    plot_off_diagonal(Aprime, INV)

                    
                    
                # A STATE WITHOUT EMERGENT COEXISTENCE / NO EXCLUDING ELEMENTS 
                else: 
                    x_data.append(S)
                    phi_data.append(spectral_radius(Aprime))             


# MATHEMATICAL PREDICTIONS OF MINIMAL AND MAXIMAL SPECTRAL RADIUS FOR EC STATES
                    
Smath = np.arange(3, Smax)
Max_Phi = np.zeros(len(Smath))
Min_Phi = np.zeros(len(Smath))
   
for i in range(len(Smath)):
    Sp = Smath[i]
    n = 1*1*Sp*(Sp-1)
    gammas = scipy.special.gamma(n/2) / scipy.special.gamma((n-1)/2)
    root = np.sqrt(1 - ( (2/(n-1))* (  gammas**2  ) ) )
    mumin = 0.071 # INFERRED FROM FIGURE 2A AS WE DO NOT HAVE AN ANALYTICAL LINE FOR THE TRANSITION TO UNBOUNDED GROWTH (see ESM II.B.3)
    muMax = -1 + (((Sp)**1.08)/14.118) * sigmamin # ANALYTICAL PREDICTION FOR MOST COMPETITIVE STATE (see ESM II.B.3)
    Max_Phi[i] = 1*(Sp-1)*abs(muMax) 
    Min_Phi[i] = (Sp - 1)*abs(mumin) 

plt.figure(figsize=(6, 5))
plt.scatter(x_data, phi_data, color='lightgray', alpha=0.01)
plt.scatter(x_data_EC, phi_data_EC, color='purple', alpha=0.1)
plt.plot(Smath, Max_Phi, linestyle="--", color="firebrick")
plt.plot(Smath, Min_Phi, linestyle="--", color="firebrick")
plt.axhline(1, color='black', linestyle='--')
plt.xlabel("S", fontsize=12)
plt.ylabel("phi", fontsize=12)
plt.tight_layout()
plt.savefig("Collectivity_Figure3A.png", format="png")# dpi=100)  # Save as SVG with high DPI
plt.show()


