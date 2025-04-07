import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import random as rd
import itertools     
import scipy.special
from scipy.optimize import root_scalar, minimize_scalar
from scipy.stats import norm
from collections import defaultdict
import seaborn as sns

# CODE TO SIMULATE AN EMPIRICAL TEST IN WHICH WE TRY TO INFER COMPOSITION FROM IMPRECISE PAIRWISE INTERACTIONS
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

error = 0.2

# SIGMA RANGE
sigmamin = 0.0
sigmamax = 1.0

# NUMBER OF SAMPLED STATES THAT WILL BE TESTED FOR COEXISTENCE, STABILITY, EMERGENT COEXISTENCE AND COLLECTIVITY
reps = 1000000

# DEFINE DATA TO FILL 
Sprime = []
Spectral = []
Coexistence = []
Stable = []

##################################################################################
##################################################################################


 
##################################################################################
##################################################################################

# MAIN A

for i in range(frac):
    
    print(i," / ", frac)
    # STATE DIVERSITY
    S = Smin + int(i*(Smax-Smin)/frac)
    
    for j in range(reps):
        
        # GENERATE RANDOM MU,SIGMA PAIR
        mu = np.random.uniform(mumin, mumax)
        sigma = np.random.uniform(sigmamin, sigmamax)
        
        # DEFINE A
        A = np.random.normal(mu, sigma, size=(80, 80))
        
        # INCORPORATE RELATIVE SELF-REGULATION
        np.fill_diagonal(A, -np.ones(80))
        
        # DEFINE A SMALL ERROR IN THE INTERACTIONS AS A PERTURBATION TO A_ij ELEMENTS
        perturbation = np.random.normal(loc=0, scale=error, size=A.shape)
        np.fill_diagonal(perturbation, np.zeros(len(A))) # perturb interactions only...
        A_perturbed = A + perturbation
        
        # SELECT A RANDOM SUBSET OF SIZE "S^*" WHICH HERE IS SIMPLY "S"
        species = np.random.choice(80, size=S, replace=False)
        
        # MEASURE INTERACTION PROPERTIES OF THIS SUBSET
        Aprime = A[np.ix_(species, species)]
        Aprime_perturbed = A_perturbed[np.ix_(species, species)]
        
        INV = np.linalg.inv(-Aprime_perturbed)
        row_sums = np.sum(INV, axis=1)
        
        # Does the perturbed subset coexist?
        
        if np.all(row_sums > 0) and np.any(Aprime < -1):
        
            # Ok, the subset coexists, is this a valid prediction?
            INV = np.linalg.inv(-Aprime)
            row_sums = np.sum(INV, axis=1)
            if np.any(row_sums <= 0):
                Coexistence.append(0) # The real case did not coexist, wrong prediction!
            else:
                Coexistence.append(1)
            
            # How does this relate to the condition number of the original matrix? here labeled spectral for consistency with previous code    
            Spectral.append(np.linalg.cond(Aprime))    
        
            Sprime.append(S)
            
            # BEYOND COEXISTENCE: PREDICTING STABILITY  
            J = np.zeros((S, S))
            for ro in range(S):
                for co in range(S):
                    J[ro, co] = row_sums[ro] * Aprime[ro, co]
            
            if np.max(np.real(np.linalg.eigvals(J))) < 0:
                Stable.append(1)
            else:
                Stable.append(0)

            
# GENERATE DIFFERENT VISUALIZATIONS TO STUDY HOW THE SUCCESS OF PREDICTING COEXISTENCE IS RELATED TO THE CONDITION NUMBER OF THE SYSTEM   
   
# IMPORT NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, linregress

# FUNCTION TO ESTIMATE THE PROBABILITY P(y=1 | x) USING KERNEL DENSITY ESTIMATION (KDE)
def kde_probability(x, y, bandwidth=0.5):
    x = np.array(x)
    y = np.array(y)
    
    # FILTER INPUT DATA TO ONLY INCLUDE INSTANCES WHERE y == 1
    x_1 = x[y == 1]

    # APPLY GAUSSIAN KERNEL DENSITY ESTIMATION TO x VALUES WHERE y == 1
    kde = gaussian_kde(x_1, bw_method=bandwidth)

    # CREATE A RANGE OF x VALUES FOR SMOOTH PLOTTING
    x_vals = np.linspace(min(x), max(x), 100)

    # CALCULATE KDE VALUES FOR THE RANGE OF x VALUES
    y_probs = kde(x_vals)

    # NORMALIZE THE OUTPUT TO SCALE PROBABILITIES BETWEEN 0 AND 1
    y_probs /= np.max(y_probs)

    # RETURN THE x VALUES AND THE NORMALIZED PROBABILITIES
    return x_vals, y_probs

# FUNCTION TO COMPUTE ROLLING MEAN OF DATA
def rolling_mean(x, y):
    """COMPUTES ROLLING MEAN WITH SORTING AND DOWNSAMPLING."""
    window = 100  # WINDOW SIZE FOR ROLLING MEAN
    step = 1      # STEP SIZE FOR ITERATION

    # SORT DATA BY x VALUES
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]

    # INITIALIZE LISTS TO STORE ROLLING MEAN VALUES
    rolling_x, rolling_y = [], []

    # LOOP THROUGH DATA TO CALCULATE ROLLING MEANS
    for i in range(0, len(x_sorted) - window + 1, step):
        rolling_x.append(np.mean(x_sorted[i:i + window]))
        rolling_y.append(np.mean(y_sorted[i:i + window]))

    # RETURN ROLLING x AND y VALUES AS ARRAYS
    return np.array(rolling_x), np.array(rolling_y)

# FUNCTION TO FIT A POLYNOMIAL TO DATA
def polynomial_fit(x, y, degree=2):
    """FITS A POLYNOMIAL TO THE ROLLING MEAN DATA."""
    coeffs = np.polyfit(x, y, degree)  # COMPUTE POLYNOMIAL COEFFICIENTS
    poly_func = np.poly1d(coeffs)      # CREATE POLYNOMIAL FUNCTION OBJECT
    x_vals = np.linspace(min(x), max(x), 100)  # CREATE x RANGE
    y_vals = poly_func(x_vals)                # COMPUTE CORRESPONDING y VALUES
    return x_vals, y_vals, poly_func          # RETURN FITTED VALUES AND FUNCTION

# FUNCTION TO PERFORM SIMPLE LINEAR REGRESSION
def linear_regression(x, y):
    slope, intercept, _, _, _ = linregress(x, y)  # PERFORM LINEAR REGRESSION
    x_vals = np.linspace(min(x), max(x), 100)     # x RANGE FOR LINE
    y_vals = slope * x_vals + intercept           # CALCULATE y VALUES FROM LINE
    return x_vals, y_vals, slope, intercept       # RETURN LINE DATA AND PARAMETERS

# CREATE A PLOTTING CANVAS WITH 3 ROWS AND 2 COLUMNS OF SUBPLOTS
fig, axes = plt.subplots(3, 2, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})

# --- FIRST SCATTER PLOT: SPECTRAL VS. COEXISTENCE ---
sc1 = axes[0, 0].scatter(Spectral, Coexistence, c=Spectral, cmap='viridis', edgecolor='k', alpha=0.1)
axes[0, 0].set_xlabel("SPECTRAL")
axes[0, 0].set_ylabel("COEXISTENCE")
axes[0, 0].set_title("SPECTRAL VS. COEXISTENCE")
cbar1 = plt.colorbar(sc1, ax=axes[0, 0])
cbar1.set_label("SPECTRAL")

# LINEAR REGRESSION FOR COEXISTENCE
x_lin, y_lin, slope, intercept = linear_regression(Spectral, Coexistence)
axes[0, 0].plot(x_lin, y_lin, color='blue', linestyle='--', label=f"LINEAR: y={slope:.3f}x+{intercept:.3f}")
axes[0, 0].legend()

# --- FIRST KDE PLOT: PREDICTED PROBABILITY OF COEXISTENCE ---
x_vals, y_probs = kde_probability(Spectral, Coexistence)
axes[1, 0].plot(x_vals, y_probs, color='red', linewidth=2)
axes[1, 0].set_xlabel("SPECTRAL")
axes[1, 0].set_ylabel("P(COEXISTENCE = 1)")
axes[1, 0].set_title("KDE ESTIMATE OF P(COEXISTENCE=1 | SPECTRAL)")
axes[1, 0].set_ylim(0, 1)

# --- ROLLING MEAN PLOT FOR COEXISTENCE ---
x_roll, y_roll = rolling_mean(Spectral, Coexistence)
axes[2, 0].plot(x_roll, y_roll, color='green', linewidth=2, marker='o', markersize=4, alpha=0.7, label="ROLLING MEAN")
axes[2, 0].set_xlabel("SPECTRAL")
axes[2, 0].set_ylabel("ROLLING MEAN")
axes[2, 0].set_title("ROLLING MEAN OF COEXISTENCE (WINDOW=50)")
axes[2, 0].set_ylim(0, 1)

# --- SECOND SCATTER PLOT: SPECTRAL VS. STABLE ---
sc2 = axes[0, 1].scatter(Spectral, Stable, c=Spectral, cmap='viridis', edgecolor='k', alpha=0.1)
axes[0, 1].set_xlabel("SPECTRAL")
axes[0, 1].set_ylabel("STABLE")
axes[0, 1].set_title("SPECTRAL VS. STABLE")
cbar2 = plt.colorbar(sc2, ax=axes[0, 1])
cbar2.set_label("SPECTRAL")

# LINEAR REGRESSION FOR STABLE (NOTE: USING Sprime INSTEAD OF Spectral HERE)
x_lin, y_lin, slope, intercept = linear_regression(Sprime, Stable)
axes[0, 1].plot(x_lin, y_lin, color='blue', linestyle='--', label=f"LINEAR: y={slope:.3f}x+{intercept:.3f}")
axes[0, 1].legend()

# --- SECOND KDE PLOT: PREDICTED PROBABILITY OF STABILITY ---
x_vals, y_probs = kde_probability(Spectral, Stable)
axes[1, 1].plot(x_vals, y_probs, color='red', linewidth=2)
axes[1, 1].set_xlabel("SPECTRAL")
axes[1, 1].set_ylabel("P(STABLE = 1)")
axes[1, 1].set_title("KDE ESTIMATE OF P(STABLE=1 | SPECTRAL)")
axes[1, 1].set_ylim(0, 1)

# ADJUST LAYOUT TO AVOID OVERLAPPING CONTENT
plt.tight_layout()

# SAVE THE FIGURE TO A FILE
plt.savefig("Figure_S9B.png", format='png')

# DISPLAY THE PLOT
plt.show()

