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

# CODE TO EXPLORE PROPERTIES OF FEASIBLE AND UNFEASIBLE INTERACTION MATRICES OF COMMUNITIES WITH EMERGENT COEXISTENCE
# GUIM AGUADE-GORGORIÓ APR 2025


####################### PARAMETERS AND SIMULATION VALUES ##############################

# DEFINE MIN AND MAX VALUES FOR THE MU PARAMETER (RANDOM NORMAL DISTRIBUTION)
mumin = -2
mumax = 0.5

# DEFINE MIN AND MAX VALUES FOR THE SIGMA PARAMETER (RANDOM NORMAL DISTRIBUTION)
sigmamin = 0.0
sigmamax = 1.0

# DEFINE THE RANGE OF SUBSET SIZES FOR THE SPECIES
subsetmin, subsetmax = 3, 13

# DEFINE NUMBER OF STATES AND REPLICAS
states = 1000000
replicas = 100

# NUMBER OF SPECIES (MATRIX SIZE)
S = 80

# GAMMA VALUE USED IN THE CALCULATION
gamma = 14.118

# FUNCTION TO SHUFFLE THE OFF-DIAGONAL ELEMENTS OF A MATRIX WHILE KEEPING THE DIAGONAL ELEMENTS IN PLACE
def shuffle_off_diagonal(matrix):
    """
    Shuffles the off-diagonal elements of a matrix without changing the diagonal elements.
    
    Parameters:
        matrix (ndarray): A square numpy array.
    
    Returns:
        ndarray: A new matrix with shuffled off-diagonal elements.
    """
    # ENSURE THE MATRIX IS SQUARE
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    
    n = matrix.shape[0]
    
    # EXTRACT OFF-DIAGONAL ELEMENTS
    off_diagonal_indices = [(i, j) for i in range(n) for j in range(n) if i != j]
    off_diagonal_values = [matrix[i, j] for i, j in off_diagonal_indices]
    
    # SHUFFLE THE OFF-DIAGONAL VALUES
    np.random.shuffle(off_diagonal_values)
    
    # CREATE A COPY OF THE ORIGINAL MATRIX
    shuffled_matrix = matrix.copy()
    
    # PLACE THE SHUFFLED VALUES BACK INTO THE OFF-DIAGONAL POSITIONS
    for (i, j), value in zip(off_diagonal_indices, off_diagonal_values):
        shuffled_matrix[i, j] = value
    
    return shuffled_matrix

# FUNCTION TO COMPUTE THE HETEROGENEITY OF THE COLUMNS IN A MATRIX
def column_heterogeneity(matrix, num_shuffles=10):
    # STEP 1: COMPUTE THE STANDARD DEVIATION OF ROW SUMS FOR THE ORIGINAL MATRIX
    row_sums = np.sum(matrix, axis=1)
    original_stddev = np.std(row_sums)
    
    # STEP 2: SHUFFLE THE MATRIX MULTIPLE TIMES AND COMPUTE THE AVERAGE STANDARD DEVIATION
    shuffled_stddevs = []
    for _ in range(num_shuffles):
        # SHUFFLE THE ENTIRE MATRIX
        shuffled_matrix = shuffle_off_diagonal(matrix)
        shuffled_row_sums = np.sum(shuffled_matrix, axis=1)
        shuffled_stddevs.append(np.std(shuffled_row_sums))
    
    # COMPUTE THE AVERAGE STANDARD DEVIATION OF THE SHUFFLED MATRICES
    average_shuffled_stddev = np.mean(shuffled_stddevs)
    
    return original_stddev / average_shuffled_stddev


##################################################################################




from concurrent.futures import ProcessPoolExecutor, as_completed

# FUNCTION TO PROCESS A SINGLE STATE AND COLLECT RESULTS
def process_state(i, S, mumin, mumax, sigmamin, sigmamax, subsetmin, subsetmax, gamma, replicas):
    # INITIALIZE DATA STRUCTURES FOR STORING RESULTS
    state_results = {'d_A': None, 'd_F': [], 'd_UF': [], 'd_FS': [], 'd_FUS': [], "d_F_phi": []}
    
    # DEFINE A STATE (RANDOMLY SELECT MU AND SIGMA, GENERATE MATRIX A)
    mu = np.random.uniform(mumin, mumax)
    sigma = np.random.uniform(sigmamin, sigmamax)
    A = np.random.normal(mu, sigma, size=(S, S))
    np.fill_diagonal(A, -np.ones(S))  # FILL DIAGONAL WITH -1 VALUES
    
    # PROCESS REPLICAS (SIMULATE MULTIPLE INSTANCES)
    for j in range(replicas):
        # FIND A RANDOM SUBSET OF SPECIES
        SUBSET = np.random.randint(subsetmin, subsetmax + 1)
        species = np.random.choice(S, size=SUBSET, replace=False)

        Aprime = A[np.ix_(species, species)]  # EXTRACT THE SUBMATRIX FOR THE SPECIES
        
        det = np.linalg.det(-Aprime)  # CALCULATE DETERMINANT OF THE NEGATIVE SUBMATRIX
            
        if det != 0 and np.any(Aprime < -1):  # FOCUS ON CASES WITH EMERGENT COEXISTENCE (EC)
            INV = np.linalg.inv(-Aprime)  # INVERSE OF THE NEGATIVE SUBMATRIX
            row_sums = np.sum(INV, axis=1)  # CALCULATE THE ROW SUMS OF THE INVERSE MATRIX
            
            if np.all(row_sums > 0):  # CHECK IF SYSTEM IS FEASIBLE
                # FEASIBLE SYSTEM
                state_results['d_F'].append(column_heterogeneity(Aprime))  # COLLECT HETEROGENEITY METRIC
                
                               
            else:
                # UNFEASIBLE SYSTEM
                state_results['d_UF'].append(column_heterogeneity(Aprime))  # COLLECT HETEROGENEITY METRIC
    
    return state_results  # RETURN THE COLLECTED RESULTS
    
# COLLECT ALL RESULTS
d_A, d_F, d_UF, d_FS, d_FUS, d_F_phi = [0], [], [], [], [], []

# USE ProcessPoolExecutor TO PARALLELIZE THE MAIN LOOP
with ProcessPoolExecutor() as executor:
    # SUBMIT TASKS TO EXECUTOR
    futures = [executor.submit(process_state, i, S, mumin, mumax, sigmamin, sigmamax, subsetmin, subsetmax, gamma, replicas) 
               for i in range(states)]
    
    # TRACK PROGRESS
    completed_tasks = 0
    total_tasks = len(futures)
    
    # COLLECT RESULTS AS TASKS COMPLETE
    for future in as_completed(futures):
        result = future.result()  # GET THE RESULT OF A COMPLETED TASK
        d_F.extend(result['d_F'])  # ADD FEASIBLE SYSTEM RESULTS
        d_UF.extend(result['d_UF'])  # ADD UNFEASIBLE SYSTEM RESULTS
        
        # UPDATE AND PRINT PROGRESS
        completed_tasks += 1
        progress_percentage = (completed_tasks / total_tasks) * 100
        print(f"Progress: {progress_percentage:.2f}%", end='\r')  # OVERWRITE THE LINE FOR REAL-TIME UPDATE 
    
            
# CALCULATE THE TOTAL NUMBER OF RESULTS
total = len(d_UF) + len(d_F) + len(d_FUS) + len(d_FS)

# PRINT RESULTS
print("UF: ", len(d_UF)*100/total, 'F: ', len(d_F)*100/total, 'F+US: ', len(d_FUS)*100/total, 'F+S: ', len(d_FS)*100/total)
        
# COMBINE RESULTS AND LABELS FOR PLOTTING
data = [d_UF, d_F]  # DATA TO PLOT (UNFEASIBLE AND FEASIBLE SYSTEMS)
labels = ['UF', 'F']  # LABELS FOR PLOT

# CREATE A FIGURE WITH TWO SIDE-BY-SIDE SUBPLOTS
fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})

from matplotlib.colors import to_rgba
palette = [to_rgba('lightgray', alpha=0.8), to_rgba('purple', alpha=0.1)]  # COLOR PALETTE

# FIRST SUBPLOT: VIOLIN PLOT
sns.violinplot(data=data, ax=axes[0], palette=['gray', 'purple'])  # INITIAL VIOLIN PLOT
axes[0].set_xticks(range(len(labels)))  # SET X-TICKS FOR CATEGORIES
axes[0].set_xticklabels(labels)  # LABEL THE X-AXIS
axes[0].axhline(y=1, color='black', linestyle='--')  # ADD HORIZONTAL LINE AT Y=1
axes[0].set_xlabel("Categories")  # X-AXIS LABEL
axes[0].set_ylabel("row sum std")  # Y-AXIS LABEL

from matplotlib.collections import PolyCollection
for collection in axes[0].collections:  # ITERATE OVER COLLECTIONS (POLYCOLLECTIONS)
    if isinstance(collection, PolyCollection):  # ENSURE IT'S A POLYCOLLECTION
        collection.set_alpha(0.6)  # SET ALPHA VALUE FOR TRANSPARENCY


# ADD CUSTOM VERTICAL LINES FROM MIN TO MAX FOR EACH CATEGORY IN THE VIOLIN PLOT
for i, category_data in enumerate(data):
    min_val = np.min(category_data)
    max_val = np.max(category_data)
    axes[0].plot([i, i], [min_val, max_val], color='black', linestyle='-', linewidth=1.5)


plt.tight_layout()  # ADJUST LAYOUT TO PREVENT OVERLAPPING
plt.savefig(f"rowsumstd_{subsetmin}_{subsetmax}.svg", format="svg", dpi=300)  # SAVE AS SVG WITH HIGH DPI
plt.show()  # DISPLAY THE PLOT




