import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import random as rd
import itertools     

# CODE TO EXPLORE THE FRACTION OF STATES THAT CARRY EC FOR TWO MU VALUES AND INCREASING SIGMA
# GUIM AGUADE-GORGORIÓ APR 2025

####################### PARAMETERS AND SIMULATION VALUES ##############################

# TWO MU VALUES (TOP AND BOTTOM PANELS)
mu_1 = -0.5
mu_2 = -1.5

# SIGMA RANGE
varAstart=0.0
varAmax=0.6

# FRACTION OF SIGMA STEPS
frac=50

# REPETITIONS IN EACH SIGMA VALUE
reps=5000

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

##################################################################################

# DEFINE DATA TO FILL
meanAlist = []
varAlist = []
Frac_cycles = np.zeros((frac, frac))
Survivors = np.zeros((frac, frac))
EC = np.zeros((frac, frac))
Frac_excl_pairs = np.zeros((frac, frac))
Frac_bistab_excl_pairs = np.zeros((frac, frac))
Frac_low_rank_exclusion = np.zeros((frac, frac))
Frac_RPS_triplets = np.zeros((frac, frac))
Frac_ones = np.zeros((frac, frac))
Rich_communities = np.zeros((frac,frac))
Fraction_small =  np.zeros((frac,frac))
Num_states_Sgeq3 =  np.zeros((frac,frac))
Num_different_states_Sgeq3 =  np.zeros((frac,frac))
vector = np.random.uniform(0,1,size=S)


z=0
while z < frac:

    print("row: ", z+1," of ",frac)
    startclock = timer()

    # DEFINE AN INCREASING VALUE FOR SIGMA AND STORE    
    varA = varAstart + (z*(varAmax-varAstart)/float(frac)) + 0.000001
    varAlist.append(varA)
    
    i=0
    while i<2:
    
        # SET MU1 OR MU2 (TOP OR BOTTOM PANELS)
        if i==0:
            meanA= mu_2
        elif i==1:
            meanA = mu_2
        
        if z==0:
            meanAlist.append(meanA)
        
        # SET COUNTERS TO ZERO        
        stable = 0
        surviving = 0 
        coexistence = 0
        emergent_coexistence = 0
        num_pairs_exclusion = 0
        lower_rank_exclusion = 0
        num_triplets = 0
        RPS_triplets = 0
        fraction_excluders = 0
        fraction_bistable_excluders = 0
        num_diff_states = 0
        list_states = []
        
        for j in range(reps):
            
            # DEFINE A
            A=np.random.normal(meanA, varA, size=(S,S)) 
            
            # INCORPORATE RELATIVE SELF-REGULATION
            np.fill_diagonal(A,-np.ones(S)) 
            
            # RECORD FRACTION OF EXCLUSIONS IN THE ORIGINAL POOL
            if np.any(A>1):
                Frac_ones[z,i] += 1/reps
            
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
                
                Frac_cycles[z,i] += 1/reps 
            
            else: # SAME STATE UP TO SI_threshold PRECISION
                
                stable +=1
                
                # COUNT AND STORE FRACTION OF SURVIVORS
                survivors = np.sum(np.array(finalstate) > alive)
                surviving +=  survivors # NUMBER OF SURVIVORS
                
                if num_diff_states == 0: # IF NO STATE HAS BEEN SEEN BEFORE, RECORD THIS ONE AS THE FIRST
                        
                    list_states.append(finalstate)
                    num_diff_states +=1
                    
                else: # IF OTHER STATES HAVE BEEN SEEN BEFORE, CHECK IF THIS HAS ALREADY BEEN SEEN OR ELSE MULTISTABILITY - NOT DISCUSSED IN THIS PAPER BUT SEE Aguadé-Gorgorió and Kéfi Journal of Physics: Complexity 2024.
                    is_new_state = False  # Assume finalstate is NOT new unless proven otherwise
                    for element in range(len(list_states)):  # Loop through all saved states
                        species_has_diff = False  # Flag to check if at least one species is different for all states
                        for species in range(S):  # Loop through all species in the saved state
                            # If at least one species is sufficiently different, we can stop comparing this state
                            if abs(finalstate[species] - list_states[element][species]) > SI_threshold:
                                species_has_diff = True
                                break  # No need to check further species for this state, since we found a difference
                        # If no species were different for this state, finalstate is not new
                        if not species_has_diff:
                            is_new_state = False
                            break  # No need to check other states, finalstate is not new
                        # If we found a species different for this state, consider finalstate as potentially new
                        is_new_state = True
                    # If at least one species is different from all states, add the new state
                    if is_new_state:
                        list_states.append(finalstate)
                        num_diff_states += 1
                        #print("New state added. Current list of different states:", list_states)
                
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
                if survivors > 2:

                    # COUNT THE CASES WITH AT LEAST 3 SPECIES... WILL THEY CONTAIN EC?
                    coexistence +=1
                    
                    # ARE THERE EXCLUSIONARY ELEMENTS (EC)?
                    if np.any(Aprime < -1):
                        emergent_coexistence += 1
    
                    #else: 
                    #    print("A state with S>2, but no EC") PRINT THE ABSENCE OF EC IN SCREEN FOR INFORMATIVE REASONS 
                           
                    # HOW MANY PAIRS INSIDE THE SURVIVING COMMUNITY ARE EXCLUDERS? FRACTION OF PAIRS - FIGURE 2C
                    pairs = 0
                    exclusive = 0
                    bistable = 0
                    
                    # MEASURE HOW MANY EXCLUSIONARY OR BISTABLE-EXCLUSIONARY INTERACTIONS
                    for x1 in range(survivors):
                        x2 = x1 + 1
                        while x2 < survivors:
                            pairs +=1 
                            if (Aprime[x1,x2] > 1 and Aprime[x2,x1] < 1 ) or (Aprime[x1,x2] < 1 and Aprime[x2,x1] > 1): # THIS PAIR SHOULD NOT BE THERE
                                exclusive += 1    
                            elif Aprime[x1,x2] > 1 and Aprime[x2,x1] > 1:
                                bistable += 1
                            x2 +=1 
                    
                    fraction_excluders += (exclusive+bistable) / pairs # I ADD ONE-WAY EXCL. AND MUTUAL EXCL.
                    fraction_bistable_excluders += bistable / pairs
                    
                    # OF ALL THE EXCLUDING PAIRS, HOW MANY TIMES A LOW-RANK SPP EXCLUDES A HIGHER-RANK ONE? (SEE FIGURE 4 AND ESM)
                    
                    wins = np.zeros(survivors)
                    loses = np.zeros(survivors)
                    
                    for x1 in range(survivors):
                        x2 = x1 + 1
                        while x2 < survivors:
                            if Aprime[x1,x2] > 1 and Aprime[x2,x1] < 1:
                                wins[x2] +=1
                                loses[x1] +=1
                            x2 +=1
                    
                    rank = (wins-loses)/pairs
                    
                    # EXCLUSION BY A LOWER RANK (TRANSITIVITY):
                    
                    for x1 in range(survivors):
                        x2 = x1 + 1
                        while x2 < survivors:
                            
                            if Aprime[x1,x2] > 1 and Aprime[x2,x1] < 1:
                                num_pairs_exclusion += 1
                            elif Aprime[x1,x2] < 1 and Aprime[x2,x1] > 1:
                                num_pairs_exclusion += 1
                                
                            if Aprime[x1,x2] > 1 and Aprime[x2,x1] < 1 and rank[x2]<rank[x1]: # 2 EXCLUDES 1 BUT HAS LOWER RANK!
                                lower_rank_exclusion +=1
                            elif Aprime[x2,x1] > 1 and Aprime[x1,x2] < 1 and rank[x1]<rank[x2]: # 1 EXCLUDES 2 BUT HAS LOWER RANK!
                                lower_rank_exclusion +=1    
                            
                            x2 +=1                    
                    
                    # PRESENCE OF R-P-S TRIPLETS (SEE FIGURE 4 AND ESM)
                    
                    # NUMBER OF EXCLUDING TRIPLETS THAT ARE INTRANSITIVE (ROCK-PAPER-SCISSORS SCHEMES)
                    for x1 in range(survivors):
                        x2 = x1 + 1
                        while x2 < survivors:
                            x3 = x1 + x2 + 1
                            while x3 < survivors:
                    
                                # IS THIS AN EXCLUDING (BUT NOT MUTUALLY EXCLUDING) TRIPLET?
                        
                                if ( (Aprime[x1,x2] > 1 and Aprime[x2,x1] < 1 ) or ( Aprime[x1,x2] < 1 and Aprime[x2,x1] > 1) ) and ( ( Aprime[x2,x3] > 1 and  Aprime[x3,x2] < 1) or ( Aprime[x2,x3] < 1 and Aprime[x3,x2] > 1) ) and ( (Aprime[x3,x1] > 1 and  Aprime[x1,x3] < 1) or ( Aprime[x3,x1] < 1 and Aprime[x1,x3] > 1) ) :
                                    num_triplets += 1
                            
                                # IS THIS A R-P-S EXCLUDING TRIPLET?
                                if Aprime[x1,x2] > 1 and Aprime[x2,x1] < 1 and Aprime[x2,x3] > 1 and Aprime[x3,x2] < 1 and Aprime[x3,x1] > 1 and Aprime[x1,x3] < 1 : 
                                    # 2 EXCLUDES 1, 3 EXCLUDES 2, 1 EXCLUDES 3
                                    RPS_triplets += 1
                       
                                x3 += 1
                            
                            x2+=1                                                    
                                
                    
                    
        
        # AVERAGE MEASURED DATA ACROSS SIMULATIONS AND STABLE STATES
        
        Num_states_Sgeq3[z,i] = coexistence
        
        Num_different_states_Sgeq3[z,i] = num_diff_states
        
        if stable > 0:
            Survivors[z,i] = surviving/stable
            if Survivors[z,i] > 3:
                Rich_communities[z,i] = 1
    
        if coexistence>0:
            EC[z,i] = emergent_coexistence / coexistence # GIVEN THERE IS COEXISTENCE, IN HOW MANY INSTANCES THERE IS EVIDENCE OF EC?
            Frac_excl_pairs[z,i] = fraction_excluders / coexistence # OF ALL THE SPECIES PAIRS, HOW MANY HAVE EXCLUSION?
            Frac_bistab_excl_pairs[z,i] = fraction_bistable_excluders / coexistence
        
        if num_pairs_exclusion > 0:
            Frac_low_rank_exclusion[z,i] = lower_rank_exclusion / num_pairs_exclusion
        
        if num_triplets > 0:
            Frac_RPS_triplets[z,i] = RPS_triplets/num_triplets
        
        i+=1
    z+=1    
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)



################## PLOTS ###############################

# PLOT 1: EC

fig,( (surviv, ec, exclusion, low_rank, triplets, states, diffstates),  (surviv2, ec2, exclusion2, low_rank2, triplets2, states2, diffstates2))= plt.subplots(2,7,figsize=(30,5))

# OBTAIN THE SIGMA_1 AND SIGMA_2 TRANSITION LINES

from scipy.special import erf
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.optimize import bisect

# Given parameters
S = 80
mu = -0.5 # MU_1

# SEARCH FOR THE SIGMA VALUE AT WHICH THE PROBABILITY OF FINDING ONE EXCLUSIONARY ELEMENT EQUALS 0.5

def probability_function(sigma):
    phi_val = norm.cdf((-1 - mu) / sigma)  # CDF of standard normal at (-1 - mu) / sigma
    return 1 - (1 - phi_val) ** (S * (S - 1))

# Define the equation to solve: P(sigma) - 0.5 = 0
def equation(sigma):
    return probability_function(sigma) - 0.5

# Use bisection method to find sigma where P(sigma) = 0.5
sigma_1 = bisect(equation, 0.01, 1)  # Search in reasonable range

print(f"Estimated sigma_1 value: {sigma_1:.5f}")


# SEARCH FOR THE SIGMA VALUE AT WHICH THE PROBABILITY OF FINDING ONE COEXISTING PAIR EQUALS 0.5

import scipy.stats as stats
import scipy.optimize as optimize

def probability_function(sigma, S=80, mu=-1.5):
    phi_val = stats.norm.cdf((-1 - mu) / sigma)
    P_C = (1 - phi_val) ** 2
    P_at_least_one_C = 1 - (1 - P_C) ** (S * (S - 1) / 2)
    return P_at_least_one_C - 0.5  # Find where this equals 0

# Use a root-finding algorithm to solve for sigma
sigma_2 = optimize.brentq(probability_function, 0.01, 1)  # Search in a reasonable range

print(f"Estimated sigma_2 value: {sigma_2:.5f}")


# PLOT


states.plot(varAlist, Num_states_Sgeq3[:,0], linestyle='-', color='b')
states.set_xlabel('varAlist')
states.set_ylabel('States')
states.set_title('states S>3, low mu')
states.axvline(x=sigma_1, color='red', linestyle='--', label=f'sigma_1^c')
states.legend()

states2.plot(varAlist, Num_states_Sgeq3[:,1], linestyle='-', color='g')
states2.set_xlabel('varAlist')
states2.set_ylabel('States')
states2.set_title('states S>3, high mu')
states2.axvline(x=sigma_2, color='red', linestyle='--', label=f'sigma_2^c')
states2.legend()

diffstates.plot(varAlist, Num_different_states_Sgeq3[:,0], linestyle='-', color='b')
diffstates.set_xlabel('varAlist')
diffstates.set_ylabel('Omega')
diffstates.set_title('Diff states S>3, low mu')
diffstates.axvline(x=sigma_1, color='red', linestyle='--', label=f'sigma_1^c')
diffstates.legend()

diffstates2.plot(varAlist, Num_different_states_Sgeq3[:,1], linestyle='-', color='g')
diffstates2.set_xlabel('varAlist')
diffstates2.set_ylabel('Omega')
diffstates2.set_title('Diff states S>3, high mu')
diffstates2.axvline(x=sigma_2, color='red', linestyle='--', label=f'sigma_2^c')
diffstates2.legend()



surviv.plot(varAlist, Survivors[:,0], linestyle='-', color='b')
surviv.set_xlabel('varAlist')
surviv.set_ylabel('Survivors')
surviv.set_title('Survivors, low mu')
surviv.axhline(y=3, color='black', linestyle='--', label='Spp rich comm')
surviv.axvline(x=sigma_1, color='red', linestyle='--', label=f'sigma_1^c')
surviv.legend()

surviv2.plot(varAlist, Survivors[:,1], linestyle='-', color='g')
surviv2.set_xlabel('varAlist')
surviv2.set_ylabel('Survivors')
surviv2.set_title('Survivors, high mu')
surviv2.axhline(y=3, color='black', linestyle='--', label='Spp rich comm')
surviv2.axvline(x=sigma_2, color='red', linestyle='--', label=f'sigma_2^c')
surviv2.legend()

ec.plot(varAlist, EC[:,0], linestyle='-', color='purple')
ec.set_xlabel('varAlist')
ec.set_ylabel('EC')
ec.set_title('At least 1 EC, low mu')
ec.axhline(y=1, color='black', linestyle='--', label='Pervasive EC')
ec.axvline(x=sigma_1, color='firebrick', linestyle='--', label=f'sigma_1^c')
#ec.legend()

ec2.plot(varAlist, EC[:,1], linestyle='-', color='purple')
ec2.set_xlabel('varAlist')
ec2.set_ylabel('EC')
ec2.set_title('At least 1 EC, high mu')
ec2.axhline(y=1, color='black', linestyle='--', label='Pervasive EC')
#ec2.axvline(x=sigma_3, color='blue', linestyle='--', label=f'sigma_2^c')
ec2.axvline(x=sigma_2, color='firebrick', linestyle='--', label=f'sigma_2^c')
#ec2.legend()

exclusion.plot(varAlist, Frac_excl_pairs[:,0], linestyle='-', color='b')
exclusion.set_xlabel('varAlist')
exclusion.set_ylabel('Excl pairs')
exclusion.set_title('Excluding pairs, low mu')
exclusion.axhline(y=0.25, color='black', linestyle='--', label='Frac coexisting excluders')
exclusion.axvline(x=sigma_1, color='red', linestyle='--', label=f'sigma_1^c')
exclusion.legend()

exclusion2.plot(varAlist, Frac_excl_pairs[:,1], linestyle='-', color='g')
exclusion2.set_xlabel('varAlist')
exclusion2.set_ylabel('Excl pairs')
exclusion2.set_title('Excluding pairs, high mu')
exclusion2.axhline(y=0.25, color='black', linestyle='--', label='Frac coexisting excluders')
exclusion2.axvline(x=sigma_2, color='red', linestyle='--', label=f'sigma_2^c')
exclusion2.legend()

low_rank.plot(varAlist, Frac_low_rank_exclusion[:,0], linestyle='-', color='b')
low_rank.set_xlabel('varAlist')
low_rank.set_ylabel('Frac low rank winners')
low_rank.set_title('Low rank winners, low mu')
low_rank.axhline(y=0.01, color='black', linestyle='--', label='Frac lo rank wins')
low_rank.axvline(x=sigma_1, color='red', linestyle='--', label=f'sigma_1^c')
low_rank.legend()


low_rank2.plot(varAlist, Frac_low_rank_exclusion[:,1], linestyle='-', color='g')
low_rank2.set_xlabel('varAlist')
low_rank2.set_ylabel('Frac low rank winners')
low_rank2.set_title('Low rank winners, high mu')
low_rank2.axhline(y=0.01, color='black', linestyle='--', label='Frac lo rank wins')
low_rank2.axvline(x=sigma_2, color='red', linestyle='--', label=f'sigma_2^c')
low_rank2.legend()


triplets.plot(varAlist, Frac_RPS_triplets[:,0], linestyle='-', color='b')
triplets.set_xlabel('varAlist')
triplets.set_ylabel('Frac RPS triplets')
triplets.set_title('RPS triplets, low mu')
triplets.axhline(y=0.0, color='black', linestyle='--', label='Frac RPS triplets')
triplets.axvline(x=sigma_1, color='red', linestyle='--', label=f'sigma_1^c')
triplets.legend()


triplets2.plot(varAlist, Frac_RPS_triplets[:,1], linestyle='-', color='b')
triplets2.set_xlabel('varAlist')
triplets2.set_ylabel('Frac RPS triplets')
triplets2.set_title('RPS triplets, high mu')
triplets2.axhline(y=0.0, color='black', linestyle='--', label='Frac RPS triplets')
triplets2.axvline(x=sigma_2, color='red', linestyle='--', label=f'sigma_2^c')
triplets2.legend()




fig.tight_layout()

plt.savefig("2_sigmavalues_FIG_2B.png", format='png')
plt.close()




exit()














