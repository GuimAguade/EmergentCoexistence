import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd     
import math
import itertools

# CODE TO EXPLORE MULTIPLE PROPERTIES OF THE GLV MODEL WITHIN THE MU-SIGMA PHASE SPACE
# GUIM AGUADE-GORGORIÓ APR 2025


####################### PARAMETERS AND SIMULATION VALUES ##############################

# MU RANGE
meanAstart=-2
meanAmax=0.5

# SIGMA RANGE
varAstart=0.0
varAmax=1.0

# GRID SIZE FRACxFRAC
frac=80

# REPETITIONS IN EACH CELL
reps=100

# SPECIES IN THE INITIAL POOL
S = 80

# TIME FRAME + ADDITIONAL WINDOW
temps1 = 3000
temps2 = 100

# THRESHOLD OF EXTINCTION
EPS=10.**-20 

# PRECISION OF ABUNDANCE COMPARISON FOR STATIONARITY
SI_threshold = 0.001

# PRECISION OF SPECIES-IS-ALIVE 
alive = 0.001


# MEASURE THE FRACTION OF MUTUAL-EXCLUSIONS VS ONE-WAY-EXCLUSIONS+MUTUAL-EXCLUSIONS    
def bistable_vs_excl_pairs(Aprime):
    n = Aprime.shape[0]
    count_both = 0  # Both A[i, j] and A[j, i] < -1
    count_at_least_one = 0  # At least one of A[i, j] or A[j, i] < -1
    
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle to avoid double-counting
            if Aprime[i, j] < -1 and Aprime[j, i] < -1:
                count_both += 1
            if Aprime[i, j] < -1 or Aprime[j, i] < -1:
                count_at_least_one += 1
    
    return count_both/ count_at_least_one if count_at_least_one > 0 else 0



# DEFINE DATA TO FILL - NOT ALL ELEMENTS ARE EXPLORED IN THIS SIMULATION - SEE ESM CODES
meanAlist = []
varAlist = []
Frac_cycles = np.zeros((frac, frac)) 
Frac_surv = np.zeros((frac, frac)) 
Presence_coex_trios = np.zeros((frac,frac))
Frac_coex_trios = np.zeros((frac,frac))
Presence_ones = np.zeros((frac, frac))
Frac_ones = np.zeros((frac, frac))
Frac_excl_pairs = np.zeros((frac, frac))
mu_ones = np.zeros((frac, frac))
sigma_ones = np.zeros((frac, frac))
Presence_coex = np.zeros((frac, frac))
Frac_coex = np.zeros((frac, frac))
Presence_onesprime = np.zeros((frac, frac))
Frac_onesprime = np.zeros((frac, frac)) 
Frac_EC = np.zeros((frac, frac))
Frac_low_rank_exclusion = np.zeros((frac, frac))
PFI = np.zeros((frac, frac))
PFI_corrected = np.zeros((frac, frac))
RPS = np.zeros((frac,frac))
BistablePairs_vs_AllExcludingPairs = np.zeros((frac, frac))




z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()
    
    # DEFINE AN INCREASING VALUE FOR SIGMA AND STORE
    varA = varAstart + (z*(varAmax-varAstart)/float(frac)) + 0.000001
    varAlist.append(varA)
    
    i=0
    while i<frac:
        
        # DEFINE AN INCREASING VALUE FOR A
        meanA= meanAstart + (i*(meanAmax-meanAstart)/float(frac))
        if z==0:
            meanAlist.append(meanA)
        
        # SET COUNTERS TO ZERO
        num_unst = 0
        surv = 0
        num_states = 0
        coexistence = 0
        num_pairs_exclusion = 0
        lower_rank_exclusion =0 
        num_coex_nou = 0
        estats_g1 = 0
        EC_states = 0
        exclusions = 0
        num_pairs_exclusion = 0
        lower_rank_exclusion = 0
        feed_frac = 0                
        
        for j in range(0,reps): # DIFFERENT SIMULATIONS AT EACH MU,SIGMA LOCATION
            
            # DEFINE A
            A=np.random.normal(meanA, varA, size=(S,S)) 
            
            # INCORPORATE RELATIVE SELF-REGULATION
            np.fill_diagonal(A,-np.ones(S))
            
            # INTEGRATE THE DYNAMICS
            def run(S, tmax=temps1, EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x))
                    dx[x<=EPS]=0 # SPECIES GOES EXTINCT BELOW THRESHOLD
                    x[x<=EPS]=0
                    return dx
                # RANDOM INITIAL CONDITIONS
                x0 = [v for v in np.random.random(S)]           
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
            time,trajectories = run(S)
            finalstate = [m for m in trajectories[:,-1]]
            
            # RUN MORE TIME AND OBSERVE IF STATE IS DIFFERENT 
            def run(S,tmax=temps2,EPS=EPS,**kwargs):
                def eqs(t,x):
                    dx = x*(1+np.dot(A,x))
                    dx[x<=EPS]=0
                    x[x<=EPS]=0
                    return dx
                # START FROM PREVIOUS FINAL STATE
                x0 = finalstate           
                sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
                time=sol.t
                trajectories=sol.y
                return time, trajectories
            timeplus,trajectoriesplus = run(S)
            finalstateplus = [m for m in trajectoriesplus[:,-1]] 
            
                        
            # IS THE STATE AT T=3000 THE SAME AT T=3100?
            
            diff = 0.0
            for spp in range(S):
                if abs ( finalstate[spp] - finalstateplus[spp] ) > SI_threshold:
                    diff += 1
                    break
                            
            if diff > 0: # DIFFERENT STATE, NOT STATIONARY
                
                num_unst+=1 
            
            else: # SAME STATE UP TO SI_threshold PRECISION
                
                # COUNT AND STORE FRACTION OF SURVIVORS
                surviving =  np.sum(np.array(finalstate) > alive) 
                surv += surviving / S
                Frac_surv[z,i]+=surviving / S
                
                # FIND SURVIVING SPECIES TO GENERATE Aprime
                xzeros = [] # A BINARY STATE VECTOR
                for spp in range(S):
                    if finalstate[spp]>alive: # if abundance is high enough, consider as survivor
                        xzeros.append(1)
                    else:
                        xzeros.append(0)    
                
                extinct = np.where(np.array(xzeros) == 0)[0]
                
                # Aprime CONTAINS ONLY THE INTERACTIONS BETWEEN SURVIVING SPECIES
                Aprime = np.delete(np.delete(A,extinct,axis=0),extinct,axis=1)
                np.fill_diagonal(Aprime, -np.ones(len(Aprime)))
                
                # IS THERE AT LEAST 3 SPECIES (PRECONDITION FOR EC)?
                if surviving > 2:
                    
                    # COUNT THE CASES WITH AT LEAST 3 SPECIES... WILL THEY CONTAIN EC?
                    num_states +=1
                    
                    # MEASURE THE FRACTION OF POSITIVE ELEMENTS IN THE NET EFFECTS MATRIX
                    
                    inverse_matrix = np.linalg.inv(-Aprime)
                    feed_frac += np.sum(inverse_matrix > 0) /  inverse_matrix.size
                    
                    # ARE THERE EXCLUSIONARY ELEMENTS (EC)?
                    if np.any(Aprime < - 1):
                        exclusions +=1
                        
                        # RECORD THE PRESENCE OF EC
                        Presence_onesprime[z,i] += 1
                        
                        # RECORD e.g. THE FRACTION OF PAIRS THAT ARE BISTABLE - MANY SIMILAR METRICS CAN BE EXTRACTED FROM Aprime, SEE ESM AND REMAINING CODES.
                        BistablePairs_vs_AllExcludingPairs[z,i] += bistable_vs_excl_pairs(Aprime)
                        
        # STORE FRACTION OF UNSTABLE RUNS                 
        Frac_cycles[z,i] = num_unst / float(reps)
        
        # STORE FRACTION OF SURVIVING SPECIES IN STABLE RUNS
        if num_unst < reps:
            Frac_surv[z,i] = Frac_surv[z,i] / (reps-num_unst)
        
        # STORE FRACTION OF EC STATES IN STATES WITH AT LEAST 3 SPECIES
                        
        if num_states > 0:
            Presence_onesprime[z,i] = Presence_onesprime[z,i]/float(num_states)
            PFI[z,i] = feed_frac/float(num_states)
        
        # STORE FRACTION OF EXCLUSIONS THAT ARE BISTABLE vs ALL EXCLUSIONS
        if exclusions > 0:
            BistablePairs_vs_AllExcludingPairs[z,i] = BistablePairs_vs_AllExcludingPairs[z,i]/float(exclusions)    
        
        i+=1
    z+=1    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)



# PLOT INTO HEATMAPS

fig, axarr = plt.subplots(5, 5, figsize=(50, 60))

# Function to format axes (to avoid repetition)
def format_ax(ax, title, cmap, data, meanAlist, varAlist):
    im = ax.imshow(data, cmap=cmap)
    stra = ["{:.3f}".format(i) for i in meanAlist]
    strb = ["{:.3f}".format(i) for i in varAlist]
    ax.set_xticks(np.arange(len(meanAlist)))
    ax.set_yticks(np.arange(len(varAlist)))
    ax.set_xticklabels(stra)
    ax.set_yticklabels(strb)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.invert_yaxis()
    ax.set_xlabel('mu')
    ax.set_ylabel('sigma')
    ax.set_title(title)
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

# First row: 2 plots, 3 empty squares
format_ax(axarr[0, 0], "Fraction of unst", "Purples", Frac_cycles, meanAlist, varAlist)
format_ax(axarr[0, 1], "Fraction of surv species", "Purples", Frac_surv, meanAlist, varAlist)

axarr[0,2].axis('off')

format_ax(axarr[0, 3], "Presence of coex trios AND S>2", "viridis", Presence_coex_trios, meanAlist, varAlist)
format_ax(axarr[0, 4], "Fraction of coex trios AND S>2", "viridis", Frac_coex_trios, meanAlist, varAlist)

# Remove unused axes (3rd, 4th, 5th in the first row)
#for i in range(2, 5):
#    axarr[0, i].axis('off')


# Second row: 5 filled squares
format_ax(axarr[1, 0], "Presence of >1", "viridis", Presence_ones, meanAlist, varAlist)
format_ax(axarr[1, 1], "Fraction of >1", "viridis", Frac_ones, meanAlist, varAlist)
format_ax(axarr[1, 2], "Bistab vs all excl pairs", "Purples", BistablePairs_vs_AllExcludingPairs, meanAlist, varAlist)
format_ax(axarr[1, 3], "Mu ones", "viridis", mu_ones, meanAlist, varAlist)
format_ax(axarr[1, 4], "Sigma ones", "viridis", sigma_ones, meanAlist, varAlist)

# Third row: 5 filled squares
format_ax(axarr[2, 0], "Presence of S>1", "viridis", Presence_coex, meanAlist, varAlist)
format_ax(axarr[2, 1], "Fraction of S>1", "viridis", Frac_coex, meanAlist, varAlist)
format_ax(axarr[2, 2], "Presence of EC", "Purples", Presence_onesprime, meanAlist, varAlist)
format_ax(axarr[2, 3], "Fraction of A'>1 elements", "viridis", Frac_onesprime, meanAlist, varAlist)
format_ax(axarr[2, 4], "Fraction of EC pairs", "viridis", Frac_EC, meanAlist, varAlist)

format_ax(axarr[3, 0], "LRE in EC", "viridis", Frac_low_rank_exclusion, meanAlist, varAlist)

format_ax(axarr[3, 2], "PFI", "Purples", PFI, meanAlist, varAlist)
format_ax(axarr[3, 3], "PFI off-diag", "viridis", PFI_corrected, meanAlist, varAlist)
format_ax(axarr[3, 1], "RPS in EC", "viridis", RPS, meanAlist, varAlist)

for i in range(3, 5):
    axarr[3, i].axis('off')
    
# Remove unused rows (3rd, 4th, and 5th rows)
for i in range(4, 5):
    for j in range(5):
        axarr[i, j].axis('off')

# Adjust layout and save the figure
fig.tight_layout()
plt.savefig("GLV_heatmaps_Figure2A.png", format='png')
plt.close()

