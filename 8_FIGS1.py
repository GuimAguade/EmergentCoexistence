import numpy as np
import matplotlib.pyplot as plt

# CODE TO EXPLORE THE PRESENCE OF COEXISTING PAIRS AND RPS TRIPLETS IN RANDOM MATRICES
# GUIM AGUADE-GORGORIÓ APR 2025

##########################################################################################################################


# FUNCTION TO CHECK IF THE MATRIX HAS ANY PAIR (i, j) SUCH THAT BOTH matrix[i,j] > -1 AND matrix[j,i] > -1
def has_pair(matrix):
    S = matrix.shape[0]  # SIZE OF THE MATRIX (NUMBER OF SPECIES OR NODES)
    for i in range(S):
        for j in range(S):
            if i != j:  # AVOID SELF-PAIRS
                if (matrix[i, j] > -1 and matrix[j, i] > -1):  # CHECK IF BOTH DIRECTIONS ARE > -1
                    return True
    return False  # RETURN FALSE IF NO SUCH PAIR IS FOUND

# FUNCTION TO CHECK IF THERE IS A ROCK-PAPER-SCISSORS (RPS) LOOP IN THE MATRIX
def has_rps_loop(matrix):
    S = matrix.shape[0]  # SIZE OF THE MATRIX
    for i in range(S):
        for j in range(S):
            for k in range(S):
                # CHECK FOR THREE DISTINCT INDICES
                if i != j and j != k and k != i:
                    # CHECK IF A LOOP i→j→k→i EXISTS WHERE ALL INTERACTIONS > -1
                    if (matrix[i, j] > -1 and matrix[j, k] > -1 and matrix[k, i] > -1):
                        return True
    return False  # RETURN FALSE IF NO RPS LOOP FOUND

# INITIAL PARAMETERS FOR SIMULATION
sigma_in = 0.0  # INITIAL STANDARD DEVIATION (σ)
sigma_fi = 0.4  # FINAL STANDARD DEVIATION (σ)
reps = 100      # NUMBER OF RANDOM MATRICES TO GENERATE PER σ VALUE

mu = -1.5       # MEAN VALUE FOR THE GAUSSIAN DISTRIBUTION
S = 80          # SIZE OF THE MATRIX (NUMBER OF SPECIES OR NODES)

# LISTS TO STORE σ VALUES AND CORRESPONDING PROBABILITIES
sigmas = []
counts_loops = []  # TO STORE PROBABILITY OF FINDING RPS LOOPS
counts_pairs = []  # TO STORE PROBABILITY OF FINDING POSITIVE PAIRS

x = sigma_in  # START σ VALUE

# LOOP OVER σ VALUES FROM sigma_in TO sigma_fi
while x < sigma_fi:
    
    sigmas.append(x)  # STORE CURRENT σ VALUE
    conta_loops = 0   # COUNTER FOR RPS LOOPS
    conta_pairs = 0   # COUNTER FOR POSITIVE PAIRS
    
    # GENERATE "reps" RANDOM MATRICES FOR CURRENT σ VALUE
    for j in range(reps):
        # GENERATE RANDOM MATRIX FROM GAUSSIAN DISTRIBUTION WITH MEAN mu AND STD x (σ)
        matrix = np.random.normal(mu, x, (S, S))
        
        # CHECK IF MATRIX HAS AN RPS LOOP
        if has_rps_loop(matrix):
            conta_loops += 1
        
        # CHECK IF MATRIX HAS AT LEAST ONE POSITIVE PAIR
        if has_pair(matrix):
            conta_pairs += 1
    
    # STORE PROBABILITIES (FRACTION OF MATRICES WHERE LOOP/PAIR WAS FOUND)
    counts_loops.append(conta_loops / reps)
    counts_pairs.append(conta_pairs / reps)
    
    # PRINT CURRENT σ VALUE AND PROBABILITIES FOR DEBUGGING/LOGGING
    print(x, conta_pairs / reps, conta_loops / reps)
    
    x += 0.001  # INCREMENT σ

# PLOT THE RESULTS
plt.figure(figsize=(8, 5))  # SET FIGURE SIZE
plt.plot(sigmas, counts_loops, color='firebrick', label='RPS Loops')  # PLOT RPS LOOP PROBABILITY
plt.plot(sigmas, counts_pairs, color='teal', label='Pairs')  # PLOT PAIR PROBABILITY
plt.xlabel(r'$\sigma$', fontsize=12)  # X-AXIS LABEL
plt.ylabel('Probability', fontsize=12)  # Y-AXIS LABEL
plt.title('Probability of Pair > -1 and RPS Loops', fontsize=14)  # TITLE
plt.show()  # DISPLAY PLOT


