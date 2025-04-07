import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd     
import math
import itertools
import seaborn as sns
from scipy.stats import kendalltau
from scipy.stats import kstest, norm
from scipy.spatial.distance import jensenshannon    
from scipy.stats import spearmanr
from scipy.stats import entropy, wasserstein_distance
from scipy.stats import skew, kurtosis

# CODE TO EXPLORE THE STABILITY METRICS OF RANDOM COMMUNITIES WITH MAY'S BOUND AND ROUTH-HURWITZ CRITERIA
# GUIM AGUADE-GORGORIÓ APR 2025

####################### PARAMETERS AND SIMULATION VALUES ##############################

# MU RANGE OF FIGURE 2A
mumin = -2.0
mumax = 0.5

# SIGMA RANGE
sigmamin = 0.0
sigmamax = 1.0

# DIVERSITY RANGE
subsetmin,subsetmax = 3,15

# NUMBER OF SAMPLED STATES THAT WILL BE TESTED FOR COEXISTENCE, STABILITY, EMERGENT COEXISTENCE AND COLLECTIVITY
states=1000000 # PART OF THE CODE THAT WILL BE PARALELLIZED
replicas=10

S = 80


gamma = 14.118


def characteristic_polynomial_coefficients(matrix):
    """
    Computes the coefficients of the characteristic polynomial of a matrix - NECESSARY FOR ROUTH-HURWITZ CRITERIA EVALUATION
    
    :param matrix: A square numpy array of shape (N, N).
    :return: A list of coefficients [(-1)^N, C1, C2, ..., CN].
    """
    # Get the coefficients of the characteristic polynomial
    coefficients = np.poly(matrix)
    
    return coefficients
    

def shuffle_off_diagonal(matrix):
    """
    Shuffles the off-diagonal elements of a matrix without changing the diagonal elements.
    
    Parameters:
        matrix (ndarray): A square numpy array.
    
    Returns:
        ndarray: A new matrix with shuffled off-diagonal elements.
    """
    # Ensure the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    
    n = matrix.shape[0]
    # Extract off-diagonal elements
    off_diagonal_indices = [(i, j) for i in range(n) for j in range(n) if i != j]
    off_diagonal_values = [matrix[i, j] for i, j in off_diagonal_indices]
    
    # Shuffle the off-diagonal values
    np.random.shuffle(off_diagonal_values)
    
    # Create a copy of the original matrix
    shuffled_matrix = matrix.copy()
    
    # Place the shuffled values back into the off-diagonal positions
    for (i, j), value in zip(off_diagonal_indices, off_diagonal_values):
        shuffled_matrix[i, j] = value
    
    return shuffled_matrix
    
    
##################################################################################




# IMPORT THE REQUIRED MODULE FOR PARALLEL PROCESSING
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function to process a single state
def process_state(i, S, mumin, mumax, sigmamin, sigmamax, subsetmin, subsetmax, gamma, replicas):
    # Initialize data structures for storing results
    
    state_results = {"stable_S": [],"stable_distance": [],"stable_Delta": [],"unstable_S": [],"unstable_distance": [],"unstable_Delta": []}
    
    # GENERATE RANDOM MU,SIGMA PAIR
    mu = np.random.uniform(mumin, mumax)
    sigma = np.random.uniform(sigmamin, sigmamax)
    
    # DEFINE A
    A = np.random.normal(mu, sigma, size=(S, S))
    
    # INCORPORATE RELATIVE SELF-REGULATION
    np.fill_diagonal(A, -np.ones(S))

    # Process replicas
    for j in range(replicas):
        
        # SELECT A RANDOM SUBSET OF SIZE "S^*" WHICH HERE IS SIMPLY "S"
        SUBSET = np.random.randint(subsetmin, subsetmax + 1)
        species = np.random.choice(S, size=SUBSET, replace=False)
        
        # MEASURE INTERACTION PROPERTIES OF THIS SUBSET
        Aprime = A[np.ix_(species, species)]
        off_diagonal_elements = Aprime[~np.eye(Aprime.shape[0], dtype=bool)]
        
        muprime = np.mean(off_diagonal_elements)
        sigmaprime = np.std(off_diagonal_elements)
        
        # CHOOSE IF MEASURING RMT PREDICTION OR SMALL-MATRIX CORRECTION
        
        # DISTANCE WITH CORRECTED THRESHOLD: MU^c - MU^*
        #distance =  abs( ((SUBSET ** 1.08) * sigmaprime / gamma) - 1) - abs( muprime )
        
        # DISTANCE WITH ORIGINAL THRESHOLD: MU^c - MU^*
        distance = abs( ( ((SUBSET/2)**0.5) * sigmaprime ) - 1 )  - abs( muprime )
            
        det = np.linalg.det(-Aprime)
            
        if det != 0:
            
            # MEASURE SPECIES ABUNDANCES: ARE SPECIES PRESENT?
            INV = np.linalg.inv(-Aprime)
            row_sums = np.sum(INV, axis=1)
            Dx = np.diag(row_sums)
            
            
            if np.all(row_sums > 0):  # Check feasibility
                
                J = np.zeros((SUBSET, SUBSET))
                for ro in range(SUBSET):
                    for co in range(SUBSET):
                        J[ro, co] = row_sums[ro] * Aprime[ro, co]
                
                # OBTAIN COEFFICIENTS
                C1 = characteristic_polynomial_coefficients(J)[1]
                C2 = characteristic_polynomial_coefficients(J)[2]
                C3 = characteristic_polynomial_coefficients(J)[3]
                #C4 = characteristic_polynomial_coefficients(J)[4]
                #C5 = characteristic_polynomial_coefficients(J)[5]
                
                # SECOND RH DETERMINANT (OR THIRD, UNCOMMENT)
                DELTA = (C1*C2 + C3) #+ (C3*C3) - (C1*C5) - (C1*C1*C4)
                
                # WE WILL RECORD ONLY STABLE COMMUNITIES, AND ASSESS IF THE RMT OR THE RH PREDICTIONS DID CORRECTLY DETERMINE THEY SHOULD BE STABLE
                if np.max(np.real(np.linalg.eigvals(J))) < 0 :
                    
                    #FIND STABLE COMMUNITIES, SEPARATE THOSE WITH EC AND THOSE WITHOUT
                    
                    if np.any(Aprime <-1) :
                    
                        state_results['stable_S'].append(SUBSET)
                        state_results['stable_Delta'].append(DELTA)
                        state_results['stable_distance'].append(distance)
                    
                    else :
                    
                        state_results['unstable_S'].append(SUBSET)
                        state_results['unstable_Delta'].append(DELTA)
                        state_results['unstable_distance'].append(distance)
                
                # AN ADDITIONAL STATEMENT COULD BE ADDED HERE TO RECORD UNSTABLE SYSTEMS TO GENERATE SUPPLEMENTARY FIGURE S11 
                # (ERASE THE LAST ELSE CONDITION AND CHANGE FOR THE FOLLOWING)
                #else:
                    #state_results['unstable_S'].append(SUBSET)
                    #state_results['unstable_Delta'].append(DELTA)
                    #state_results['unstable_distance'].append(distance)
                          
    return state_results



# INITIALIZE LISTS TO COLLECT RESULTS FROM ALL PROCESSES
sS, sd, sD, usS, usd, usD = [], [], [], [], [], []

# USE PROCESSPOOLEXECUTOR TO PARALLELIZE THE MAIN SIMULATION LOOP
with ProcessPoolExecutor() as executor:
    
    # SUBMIT TASKS TO THE EXECUTOR FOR EACH STATE
    futures = [
        executor.submit(
            process_state, i, S, mumin, mumax, sigmamin, sigmamax, subsetmin, subsetmax, gamma, replicas
        ) for i in range(states)
    ]
    
    # INITIALIZE A COUNTER TO TRACK COMPLETED TASKS
    completed_tasks = 0
    total_tasks = len(futures)  # TOTAL NUMBER OF TASKS

    # ITERATE OVER THE COMPLETED FUTURES AS THEY FINISH
    for future in as_completed(futures):
        result = future.result()  # GET THE RESULT FROM THE FUTURE

        # EXTEND THE GLOBAL RESULT LISTS WITH DATA FROM THIS RESULT
        sS.extend(result['stable_S'])            # STABLE SYSTEM S VALUES
        sd.extend(result['stable_distance'])     # STABLE SYSTEM DISTANCE VALUES (mu_c - muprime)
        sD.extend(result['stable_Delta'])        # STABLE SYSTEM DELTA VALUES
        usS.extend(result['unstable_S'])         # UNSTABLE SYSTEM S VALUES
        usd.extend(result['unstable_distance'])  # UNSTABLE SYSTEM DISTANCE VALUES
        usD.extend(result['unstable_Delta'])     # UNSTABLE SYSTEM DELTA VALUES

        # UPDATE AND PRINT PROGRESS IN REAL-TIME
        completed_tasks += 1
        progress_percentage = (completed_tasks / total_tasks) * 100
        print(f"Progress: {progress_percentage:.2f}%", end='\r')  # OVERWRITE THE SAME LINE FOR PROGRESS DISPLAY

# ------------------- PLOTTING RESULTS FOR (mu_c - muprime) -------------------

# CREATE A NEW FIGURE FOR THE FIRST SCATTER PLOT
plt.figure(figsize=(6, 5))

# PLOT UNSTABLE SYSTEMS WITH VERY LOW OPACITY (ALPHA) FOR VISUAL DENSITY
plt.scatter(usS, usd, color='lightgray', alpha=0.004)

# PLOT STABLE SYSTEMS WITH SLIGHTLY HIGHER OPACITY
plt.scatter(sS, sd, color='purple', alpha=0.01)

# ADD A HORIZONTAL LINE AT y=0 TO SEPARATE POSITIVE/NEGATIVE DISTANCES
plt.axhline(0, color='black', linestyle='--')

# LABEL THE AXES
plt.xlabel("S", fontsize=12)
plt.ylabel("mu_c - muprime", fontsize=12)

# OPTIMIZE LAYOUT
plt.tight_layout()

# SAVE THE PLOT AS A PNG IMAGE FILE
plt.savefig("RMT_prediction.png", format="png")  # YOU CAN ADD dpi=100 FOR HIGHER QUALITY

# DISPLAY THE PLOT
plt.show()

# ------------------- PLOTTING RESULTS FOR Delta_3 -------------------

# CREATE A NEW FIGURE FOR THE SECOND SCATTER PLOT
plt.figure(figsize=(6, 5))

# PLOT UNSTABLE SYSTEMS DELTA VALUES
plt.scatter(usS, usD, color='lightgray', alpha=0.004)

# PLOT STABLE SYSTEMS DELTA VALUES
plt.scatter(sS, sD, color='purple', alpha=0.01)

# ADD A HORIZONTAL LINE AT y=0
plt.axhline(0, color='black', linestyle='--')

# LABEL THE AXES
plt.xlabel("S", fontsize=12)
plt.ylabel("Delta_3", fontsize=12)

# OPTIONAL: LIMITS CAN BE UNCOMMENTED TO ADJUST VIEWPORT
# plt.xlim(-0.3, None)
# plt.ylim(-0.003, None)

# OPTIMIZE LAYOUT
plt.tight_layout()

# SAVE THE PLOT AS A PNG IMAGE FILE
plt.savefig("RH_prediction.png", format="png")  # YOU CAN ADD dpi=100 FOR HIGHER QUALITY

# DISPLAY THE PLOT
plt.show()

# FIGURE S10 - AN EXACT REPLICA REQUIRES UNCOMMENTING CODE ABOVE FOR MEASURING UNSTABLE STATE PROPERTIES.

# ------------------- PLOTTING mu_c - muprime vs. Delta_3 -------------------

# CREATE A NEW FIGURE FOR THE FINAL COMPARISON PLOT
plt.figure(figsize=(6, 5))

# PLOT UNSTABLE SYSTEMS: X = mu_c - muprime (usd), Y = Delta_3 (usD)
plt.scatter(usd, usD, color='lightgray', alpha=0.004)

# PLOT STABLE SYSTEMS: X = mu_c - muprime (sd), Y = Delta_3 (sD)
plt.scatter(sd, sD, color='purple', alpha=0.01)

# ADD HORIZONTAL AND VERTICAL LINES AT ZERO FOR REFERENCE
plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')

# LABEL THE AXES TO REFLECT THE TWO METRICS
plt.xlabel("mu_c - muprime", fontsize=12)
plt.ylabel("Delta_3", fontsize=12)

# SET A TITLE TO INDICATE WHAT'S BEING COMPARED
plt.title("Comparison of muprime and Delta Metrics", fontsize=14)

# OPTIMIZE LAYOUT FOR CLEAN PRESENTATION
plt.tight_layout()

# SAVE THE FIGURE TO FILE
plt.savefig("muprime_vs_delta3.png", format="png")  # YOU CAN ADD dpi=100 FOR HIGHER QUALITY

# DISPLAY THE PLOT
plt.show()





        
        


