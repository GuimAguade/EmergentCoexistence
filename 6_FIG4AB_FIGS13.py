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

# CODE TO EXPLORE THE FRACTION OF LOW RANK EXCLUSIONS IN COMMUNITIES WITH EMERGENT COEXISTENCE
# GUIM AGUADE-GORGORIÓ APR 2025

##########################################################################################################################

# GIVE A VALUE TO THE REPLICA IN CASE MANY REPLICAS ARE GENERATED IN PARALLEL TO IMPROVE DATA QUALITY
COPY = 1

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

# NUMBER OF SAMPLED STATES THAT WILL BE TESTED FOR COEXISTENCE, STABILITY, EMERGENT COEXISTENCE AND INTRANSITIVITY
reps = 1000000

# DEFINE DATA TO FILL FOR RANDOM, SHUFFLED, NESTED AND CROSS-FEEDING INTERACTIONS
# TO MAINTAIN CODING SIMPLICITY WE USE THE SAME LISTS AS FOR THE RPS MODEL
# TRI REFERS TO THE NUMBER OF EXCLUSIONS, NOT TRIPLETS
# RPS REFERS TO THE NUMBER OF LOW RANK EXCLUSIONS, NO RPS TRIPLETS
Sprime = []
TRIprime = []
RPSprime = []
Sshuff = []
TRIshuff = []
RPSshuff = []
Scross = []
TRIcross = []
RPScross = []
Snested = []
TRInested = []
RPSnested = []

##################################################################################
##################################################################################

# COUNT THE NUMBER OF EXCLUSIONARY PAIRS
def count_exclusions(matrix):
    
    # WE FOCUS ON ONE-WAY EXCLUSIONS IN WHICH EXPERIMENTAL OUTCOMES OF EXCLUSION WOULD BE WELL-CHARACTERIZED, ALTHOUGH OMITTING THIS DOES NOT YIELD ANY SIGNIFICANT DIFFERENCE
    num_species = matrix.shape[0]
    exclusions = 0
    for i in range(num_species):
        for j in range(i+1,num_species):
            if i != j:
                if matrix[i, j] > -1 and matrix[j,i]<-1:
                    exclusions += 1
                elif matrix[i, j] < -1 and matrix[j,i]>-1:
                    exclusions += 1

    return exclusions                    

# COUNT THE NUMBER OF EXCLUSIONARY PAIRS THAT ARE "LOW-RANK EXCLUDES HIGH-RANK"

def count_LRE(matrix):
    num_species = matrix.shape[0]
    ranks = np.zeros(num_species)
    violations = 0
    
    # DEFINE THE RANK OF A SPECIES BY THE NUMBER OF WINS - NUMBER OF LOSSES. COULD ALSO BE RELATIVE TO NUMBER OF INTERACTIONS
    wins = np.zeros(num_species)
    loses = np.zeros(num_species)
    for i in range(num_species):
        for j in range(i+1,num_species):
            if matrix[i,j] < -1 and matrix[j,i] >- 1: # j wins
                wins[j] +=1
                loses[i] +=1
            elif matrix[i,j] >- 1 and matrix[j,i] < -1: # i wins
                wins[i] +=1
                loses[j] +=1

    ranks=[]                
    for i in range(num_species):
        ranks.append(wins[i]-loses[i]) # INCLUDE NUMBER OF INTERACTIONS HERE TO MAKE IT RELATIVE, ALTHOUGH IN FULLY-CONNECTED MATRICES THIS IS IRRELEVANT AS ALL SPECIES HAVE Sx(S-1) INTERACTIONS
        
    # COUNT IN HOW MANY INSTANCES THERE IS A VIOLATION OF THE HIERARCHY IN WHICH THE LOWER RANKED EXCLUDES THE HIGHER RANKED
    for i in range(num_species):
        for j in range(i+1,num_species):
            if matrix[i,j]<-1 or matrix[j,i]<-1:
                if matrix[i, j] <-1 and matrix[j,i]>-1 and ranks[i] > ranks[j]:
                    violations += 1
                elif matrix[j,i] <-1 and matrix[i,j]>-1 and ranks[j] > ranks[i]:
                    violations += 1
                
    return violations

# SHUFFLE OFF-DIAGONAL ELEMENTS OF A MATRIX, WILL BE USED TO STUDY THE NULL-EXPECTATION FROM A*
def shuffle_off_diagonal(matrix):
    
    num_species = matrix.shape[0]
    
    shuffled_matrix = matrix.copy()
    
    for i in range(num_species):
        for j in range(i+1, num_species):
            if np.random.rand() < 0.5: # Switch positions
                shuffled_matrix[i,j] = matrix[j,i]
                shuffled_matrix[j,i] = matrix[i,j]
    
    return shuffled_matrix

 
##################################################################################


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
                
        # DOES THIS SUBSET LEAD TO A FEASIBLE STATE OF POSITIVE ABUNDANCES AND EMERGENT COEXISTENCE?
        INV = np.linalg.inv(-Aprime)
        row_sums = np.sum(INV, axis=1)
       
        if np.all(row_sums > 0) and np.any(Aprime < -1):
            # IF SO, CHECK FOR STABILITY
            J = np.zeros((S, S))
            for ro in range(S):
                for co in range(S):
                    J[ro, co] = row_sums[ro] * Aprime[ro, co]
            
            excl = count_exclusions(Aprime)
            
            # ARE ALL EIGENVALUES NEGATIVE (LINEARLY STABLE STATE)              
            if np.max(np.real(np.linalg.eigvals(J))) < 0 and excl > 3: # INCORPORATE THIS LAST CONDITION TO EXPLORE THE LRE STATISTICS IN COMMUNITIES WITH MORE THAN 3 EXCLUSIONS AS DISCUSSED IN THE ESM.
                
                # TRANSITIVITY METRICS FOR EC STATES
                Sprime.append(S)
                TRIprime.append(count_exclusions(Aprime))
                RPSprime.append(count_LRE(Aprime))
                
                # TRANSITIVITY METRICS FOR STATES WITH SHUFFLED (RANDOMIZED) MATRIX
                Sshuff.append(S)
                TRIshuff.append(count_exclusions(shuffle_off_diagonal(Aprime)))
                RPSshuff.append(count_LRE(shuffle_off_diagonal(Aprime)))

        # DEFINE A
        A = np.random.normal(mu, sigma, size=(80, 80))
        
        # INCORPORATE PURELY TRIANGULAR STRUCTURE
        for row in range(80):
            for col in range(row+1,80):
                A[row,col] = 0
        
        # INCORPORATE RELATIVE SELF-REGULATION
        np.fill_diagonal(A, -np.ones(80))
        
        # SELECT A RANDOM SUBSET OF SIZE "S^*" WHICH HERE IS SIMPLY "S"
        species = np.random.choice(80, size=S, replace=False)
        
        # MEASURE INTERACTION PROPERTIES OF THIS SUBSET
        Aprime = A[np.ix_(species, species)]
        off_diagonal_elements = Aprime[~np.eye(Aprime.shape[0], dtype=bool)]
                
        # DOES THIS SUBSET LEAD TO A FEASIBLE STATE OF POSITIVE ABUNDANCES AND EMERGENT COEXISTENCE?
        INV = np.linalg.inv(-Aprime)
        row_sums = np.sum(INV, axis=1)
       
        if np.all(row_sums > 0) and np.any(Aprime < -1):
            # IF SO, CHECK FOR STABILITY
            J = np.zeros((S, S))
            for ro in range(S):
                for co in range(S):
                    J[ro, co] = row_sums[ro] * Aprime[ro, co]
            
            # ARE ALL EIGENVALUES NEGATIVE (LINEARLY STABLE STATE)              
            if np.max(np.real(np.linalg.eigvals(J))) < 0:
                
                # TRANSITIVITY METRICS FOR NESTED STATES
                Snested.append(S)
                TRInested.append(count_exclusions(Aprime))
                RPSnested.append(count_LRE(Aprime))
                
                
##################################################################################

# STUDY INTRANSITIVITY IN SYSTEMS WITH RESOURCE COMPETITION + CROSS-FEEDING


for i in range(frac):
    
    print(i," / ", frac)
    S = Smin + int(i*(Smax-Smin)/frac)
    
    for j in range(reps):
        
        # GENERATE GAMMA VALUES
        gamma = np.sort(np.random.uniform(0.3,0.7,size=80))

        # PARAMETERS FOR THE WEAK RANDOM COMPETITION INTERACTIONS
        amplify = 0.1 #* np.random.rand() # WE CAN ADD A RANDOM COMPONENT TO THE STRENGTH AND HETEROGENEITY OF INTERACTIONS - NOT NECESSARY
        sigma_amplify = 0.01 #* np.random.rand()
        Connectance = 1 # np.random.uniform(0.1, 1) # CONNECTANCE CAN BE TUNED HERE

        # Create row and column index matrices
        gamma_row = gamma[:, np.newaxis]  # Shape (50,1)
        gamma_col = gamma[np.newaxis, :]  # Shape (1,50)

        # Compute B using broadcasting
        B = 2 * gamma_col / (gamma_row + gamma_col) + np.random.normal(amplify, sigma_amplify, size=(80, 80))

        # Generate C as a matrix instead of element-wise
        C = np.abs(np.random.uniform(0.0,1.0,size=(80,80)))

        # Compute A directly
        A = C - B

        # Apply Connectance mask efficiently
        # mask = np.random.rand(80, 80) < (1 - Connectance)
        # A[mask] = 0

        # INCORPORATE RELATIVE SELF-REGULATION
        np.fill_diagonal(A, -np.ones(80))
        
        # SELECT A RANDOM SUBSET OF SIZE "S^*" WHICH HERE IS SIMPLY "S"
        species = np.random.choice(80, size=S, replace=False)
        
        # MEASURE INTERACTION PROPERTIES OF THIS SUBSET
        Aprime = A[np.ix_(species, species)]
        off_diagonal_elements = Aprime[~np.eye(Aprime.shape[0], dtype=bool)]
                
        # DOES THIS SUBSET LEAD TO A FEASIBLE STATE OF POSITIVE ABUNDANCES AND EMERGENT COEXISTENCE?
        INV = np.linalg.inv(-Aprime)
        row_sums = np.sum(INV, axis=1)
       
        if np.all(row_sums > 0) and np.any(Aprime < -1):
            # IF SO, CHECK FOR STABILITY
            J = np.zeros((S, S))
            for ro in range(S):
                for co in range(S):
                    J[ro, co] = row_sums[ro] * Aprime[ro, co]
            
            excl = count_exclusions(Aprime)
            
            # ARE ALL EIGENVALUES NEGATIVE (LINEARLY STABLE STATE)              
            if np.max(np.real(np.linalg.eigvals(J))) < 0 and excl > 3: # INCORPORATE THIS LAST CONDITION TO EXPLORE THE LRE STATISTICS IN COMMUNITIES WITH MORE THAN 3 EXCLUSIONS AS DISCUSSED IN THE ESM.
                
                # TRANSITIVITY METRICS FOR RC+CF STATES
                Scross.append(S)
                TRIcross.append(count_exclusions(Aprime))
                RPScross.append(count_LRE(Aprime))
                
                
# STORE DATA IN DICTIONARIES - WILL BE USEFUL TO EXTRACT DATA FOR A SPECIFIC DIVERSITY METRIC FOR FIGURES 4B AND 4D

# Dictionary to store {Sprime_value: list of (RPSprime / TRIprime)}
grouped_ratios_prime = {}
grouped_ratios_cross = {}
grouped_ratios_shuff = {}
grouped_ratios_nested = {}

# Populate the dictionary
for s, tri, rps in zip(Sprime, TRIprime, RPSprime):
    if s not in grouped_ratios_prime:
        grouped_ratios_prime[s] = []
    grouped_ratios_prime[s].append(rps / tri if tri != 0 else np.nan)  # Avoid division by zero

# Populate the dictionary
for s, tri, rps in zip(Scross, TRIcross, RPScross):
    if s not in grouped_ratios_cross:
        grouped_ratios_cross[s] = []
    grouped_ratios_cross[s].append(rps / tri if tri != 0 else np.nan)  # Avoid division by zero

# Populate the dictionary
for s, tri, rps in zip(Sshuff, TRIshuff, RPSshuff):
    if s not in grouped_ratios_shuff:
        grouped_ratios_shuff[s] = []
    grouped_ratios_shuff[s].append(rps / tri if tri != 0 else np.nan)  # Avoid division by zero

# Populate the dictionary
for s, tri, rps in zip(Snested, TRInested, RPSnested):
    if s not in grouped_ratios_nested:
        grouped_ratios_nested[s] = []
    grouped_ratios_nested[s].append(rps / tri if tri != 0 else np.nan)  # Avoid division by zero

#import csv

#def write_to_csv(data_dict, filename):
    # Get all Sprime values (sorted)
#    sorted_sprime = sorted(data_dict.keys())
    # Create the CSV file
#    with open(filename, mode='w', newline='') as file:
#        writer = csv.writer(file)
        # Write the header row (Sprime values)
#        writer.writerow(sorted_sprime)
        # Write the corresponding rps/tri values for each Sprime
        # Make sure each column (for each Sprime) has the same length by padding with NaN if necessary
#        max_len = max(len(data_dict[s]) for s in sorted_sprime)
        # Fill the rows with the rps/tri values
#        for i in range(max_len):
#            row = []
#            for s in sorted_sprime:
                # Take the i-th element from the list or NaN if it's out of bounds
#                row.append(data_dict[s][i] if i < len(data_dict[s]) else float('nan'))
#            writer.writerow(row)

# Write the grouped ratios for each dataset to CSV - This is useful if multiple replicas are performed to generate larger datasets in parallel

#write_to_csv(grouped_ratios_prime, f'grouped_ratios_prime_{COPY}.csv')
#write_to_csv(grouped_ratios_cross, f'grouped_ratios_cross_{COPY}.csv')
#write_to_csv(grouped_ratios_shuff, f'grouped_ratios_shuff_{COPY}.csv')
#write_to_csv(grouped_ratios_nested, f'grouped_ratios_nested_{COPY}.csv')


# COMPUTE MEAN AND STD TO PLOT

S_values_prime = []
mean_values_prime = []
std_values_prime = []

S_values_cross = []
mean_values_cross = []
std_values_cross = []

S_values_shuff = []
mean_values_shuff = []
std_values_shuff = []

S_values_shuff = []
mean_values_shuff = []
std_values_shuff = []

S_values_nested = []
mean_values_nested = []
std_values_nested = []

for s in sorted(grouped_ratios_prime.keys()):  # Ensure x-axis is sorted
    ratios = np.array(grouped_ratios_prime[s])
    ratios = ratios[~np.isnan(ratios)]  # Remove NaN values
    if len(ratios) > 0:  # Avoid empty lists
        S_values_prime.append(s)
        mean_values_prime.append(np.mean(ratios))
        std_values_prime.append(np.std(ratios))

for s in sorted(grouped_ratios_cross.keys()):  # Ensure x-axis is sorted
    ratios = np.array(grouped_ratios_cross[s])
    ratios = ratios[~np.isnan(ratios)]  # Remove NaN values
    if len(ratios) > 0:  # Avoid empty lists
        S_values_cross.append(s)
        mean_values_cross.append(np.mean(ratios))
        std_values_cross.append(np.std(ratios))
        
for s in sorted(grouped_ratios_shuff.keys()):  # Ensure x-axis is sorted
    ratios = np.array(grouped_ratios_shuff[s])
    ratios = ratios[~np.isnan(ratios)]  # Remove NaN values
    if len(ratios) > 0:  # Avoid empty lists
        S_values_shuff.append(s)
        mean_values_shuff.append(np.mean(ratios))
        std_values_shuff.append(np.std(ratios))       

for s in sorted(grouped_ratios_nested.keys()):  # Ensure x-axis is sorted
    ratios = np.array(grouped_ratios_nested[s])
    ratios = ratios[~np.isnan(ratios)]  # Remove NaN values
    if len(ratios) > 0:  # Avoid empty lists
        S_values_nested.append(s)
        mean_values_nested.append(np.mean(ratios))
        std_values_nested.append(np.std(ratios)) 



# FIGURE 4C

plt.figure(figsize=(8, 5))

plt.plot(S_values_prime, mean_values_prime, "o-", label="Mean (with ±2 Std Dev)", color="purple")
plt.fill_between(S_values_prime,np.array(mean_values_prime) - 0.25 * np.array(std_values_prime),np.array(mean_values_prime) + 0.25 * np.array(std_values_prime),color="purple",alpha=0.2)

plt.plot(S_values_nested, mean_values_nested, "o-", label="Mean (with ±2 Std Dev)", color="firebrick")
plt.fill_between(S_values_nested,np.array(mean_values_nested) - 0.25 * np.array(std_values_nested),np.array(mean_values_nested) + 0.25 * np.array(std_values_nested),color="firebrick",alpha=0.2)

plt.plot(S_values_cross, mean_values_cross, "o-", label="Mean (with ±2 Std Dev)", color="teal")
plt.fill_between(S_values_cross,np.array(mean_values_cross) - 0.25 * np.array(std_values_cross),np.array(mean_values_cross) + 0.25 * np.array(std_values_cross),color="teal",alpha=0.2)

plt.plot(S_values_shuff, mean_values_shuff, "o-", label="Mean (with ±2 Std Dev)", color="gray")
plt.fill_between(S_values_shuff,np.array(mean_values_shuff) - 0.25 * np.array(std_values_shuff),np.array(mean_values_shuff) + 0.25 * np.array(std_values_shuff),color="gray",alpha=0.2)

plt.axhline(y=0, color="black", linestyle="--", linewidth=1, label="y = 0")

plt.xlabel("Species Diversity")
plt.ylabel("Mean and Std Dev of LRE fraction")
plt.savefig(f"Fig4A.png", format='png', dpi=200)
plt.show()




#########################################################################################################################

# FIGURE 4B 

# SET SPECIES DIVERSITY AT WHICH WE WANT TO GET THE KDE

element = 6


# Find the column index where the S_value is 7 or 8 or whatever we impose in "element"...

if element in grouped_ratios_prime:
    values_S7_prime = grouped_ratios_prime[element]
        
if element in grouped_ratios_cross:
    values_S7_cross = grouped_ratios_cross[element]

if element in grouped_ratios_shuff:
    values_S7_shuff = grouped_ratios_shuff[element]    

if element in grouped_ratios_nested:
    values_S7_nested = grouped_ratios_nested[element]
        

data_cross = values_S7_cross
data_prime = values_S7_prime
data_shuff = values_S7_shuff
data_nested = values_S7_nested

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Clean function to convert any "nan" strings to np.nan and keep only numeric values
def clean_data(data):
    # Convert "nan" or any non-numeric to np.nan and convert to float
    data_cleaned = pd.to_numeric(data, errors='coerce')
    # Remove NaN values using numpy
    data_cleaned = data_cleaned[~np.isnan(data_cleaned)]  # ~np.isnan filters out NaNs
    return data_cleaned

# Clean all datasets
data_cross_clean = clean_data(data_cross)
data_prime_clean = clean_data(data_prime)
data_shuff_clean = clean_data(data_shuff)
data_nested_clean = clean_data(data_nested)

# Step 1: Generate 100 subsets with 100 random elements each - This is a key step: in experiments such as Chang et al 2023, the statistics of RPS are studied over a large number of observed triplets. to do so, here we replicate his experiment by assuming what would happen if we observed 100 systems with 100 triplets each, making the statistics much more robust to variability.

subset_size = 100
n_subsets = 100

# Generate subsets and calculate the mean of each subset
subset_means_cross = []
subset_means_shuff = []
subset_means_prime = []
subset_means_nested = []

for _ in range(n_subsets):
    
    subset = np.random.choice(data_cross_clean, size=subset_size, replace=True)
    subset_means_cross.append(np.mean(subset))
    
    subset = np.random.choice(data_prime_clean, size=subset_size, replace=True)
    subset_means_prime.append(np.mean(subset))
    
    subset = np.random.choice(data_shuff_clean, size=subset_size, replace=True)
    subset_means_shuff.append(np.mean(subset))
    
    subset = np.random.choice(data_nested_clean, size=subset_size, replace=True)
    subset_means_nested.append(np.mean(subset))

subset_means_cross = np.array(subset_means_cross)
subset_means_shuff = np.array(subset_means_shuff)
subset_means_prime = np.array(subset_means_prime)
subset_means_nested = np.array(subset_means_nested)

# Step 2: Plot the Gaussian Distribution for the means of the subsets
mean_of_means_cross = np.mean(subset_means_cross)
mean_of_means_shuff = np.mean(subset_means_shuff)
mean_of_means_prime = np.mean(subset_means_prime)
mean_of_means_nested = np.mean(subset_means_nested)

std_of_means_cross = np.std(subset_means_cross)
std_of_means_prime = np.std(subset_means_prime)
std_of_means_shuff = np.std(subset_means_shuff)
std_of_means_nested = np.std(subset_means_nested)

# Print in terminal de data for mean values that will be plotted in the figure by hand

print(mean_of_means_cross)
print(mean_of_means_shuff)
print(mean_of_means_prime)
print(mean_of_means_nested)

"""
# Plot the KDE for subset means without the bars
sns.histplot(subset_means_cross, kde=True, stat="density", bins=15, color='teal', label="Subset Means (Cross)", hist=False)
sns.histplot(subset_means_shuff, kde=True, stat="density", bins=15, color='gray', label="Subset Means (Shuff)", hist=False)
sns.histplot(subset_means_prime, kde=True, stat="density", bins=15, color='purple', label="Subset Means (Prime)", hist=False)
"""

# Plot only the KDE curve for each dataset (no histogram bars) and shade the area under the curve
sns.kdeplot(subset_means_cross, color='teal', label="Subset Means (Cross)", fill=True, alpha=0.5)
sns.kdeplot(subset_means_shuff, color='gray', label="Subset Means (Shuff)", fill=True, alpha=0.5)
sns.kdeplot(subset_means_prime, color='purple', label="Subset Means (Prime)", fill=True, alpha=0.5)
sns.kdeplot(subset_means_nested, color='firebrick', label="Subset Means (Nested)", fill=True, alpha=0.5)

# Add vertical dashed lines for the means
plt.axvline(mean_of_means_cross, color='teal', linestyle='--', label=f'Mean Cross: {mean_of_means_cross:.2f}')
plt.axvline(mean_of_means_shuff, color='gray', linestyle='--', label=f'Mean Shuff: {mean_of_means_shuff:.2f}')
plt.axvline(mean_of_means_prime, color='purple', linestyle='--', label=f'Mean Prime: {mean_of_means_prime:.2f}')
plt.axvline(mean_of_means_nested, color='firebrick', linestyle='--', label=f'Mean Nested: {mean_of_means_nested:.2f}')

plt.savefig(f"Fig4B.png", format='png', dpi=200)
plt.show()











