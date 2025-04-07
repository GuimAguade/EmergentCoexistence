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
import scipy.stats as stats

# CODE TO EXPLORE THE FRACTION OF EXCLUDING PAIRS IN STABLE COMMUNITIES AS A FUNCTION OF THEIR DIVERSITY
# GUIM AGUADE-GORGORIÓ APR 2025

####################### PARAMETERS AND SIMULATION VALUES ##############################

# MU RANGE
meanAstart=-2
meanAmax=0.5

# SIGMA RANGE
varAstart=0.0
varAmax=1.0

# DIFFERENT RANDOM MU,SIGMA ELEMENTS STUDIED
systems=100000

# REPETITIONS FOR EACH SYSTEM
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


##################################################################################


# EQUATION FOR THE MAX DIVERSITY (HERE S_max = x) GIVEN MU, SIGMA (SEE ESM II.A)

def search_maxS(x, mu, sigma):
    # Avoid division by zero or negative values in sqrt
    if x <= 1 or x*(x-1) <= 0:
        return 0
    else:
        return mu - (4*sigma/np.sqrt(x*(x-1))) + ((x**1.08 / 14.8) * sigma* (1- np.sqrt(1 - ((2/(x*(x-1)-1))* (scipy.special.gamma(x*(x-1)/2) / scipy.special.gamma((x*(x-1)-1)/2))*(scipy.special.gamma(x*(x-1)/2) / scipy.special.gamma((x*(x-1)-1)/2))   )))) - 1


# GIVEN SEARCH_MAXS ABOVE, FIND THE BEST INTEGER FIT THAT SOLVES IT

def find_integer_solution(mu, sigma, x_range=(2, 50), step=0.1):
    # Initialize variables to track the best solution
    best_x = None
    min_abs_Fx = float('inf')
    
    # Iterate through possible values of x in the given range with the specified step size
    for x in np.arange(x_range[0], x_range[1], step):
        Fx = search_maxS(x, mu, sigma)
        if abs(Fx) < min_abs_Fx:
            min_abs_Fx = abs(Fx)
            best_x = x
    
    # Return the integer solution if it is found within a reasonable tolerance
    return int(best_x) if best_x is not None else 0

# COUNT FRACTION OF EXCLUDING PAIRS IN A GIVEN MATRIX

def count_greater_than_one_pairs(A):
    n = A.shape[0]
    count = 0
    for i in range(n):
        for j in range(i+1, n):  # Only consider i < j to avoid double counting
            if A[i, j] <-1 or A[j, i] <-1:
                count += 1
    
    pairs = n*(n-1)/2
    
    fraction = count / pairs
    
    return fraction


# LISTS TO STORE MEASUREMENTS

coex_list = []
ec_list = []
surv_list = []
number_exclusions = []
fraction_exclusions = []
ecpairs_list =[]
number_exclusions_prime = []
fraction_exclusions_prime = []
mulist = []
sigmalist = []

# DOMAIN OF DIVERSITIES WITHIN WHICH WE WANT TO FIND THE MAX FRACTION OF EXCLUDING PAIRS ANALYTICALLY

maxS = list(range(3, 18))
maxECpairs = np.zeros(len(maxS))


##################################################################################


for z in range(systems):

    print("row: ", z+1," of ",systems)
    startclock = timer()
    
    # ZOOM IN INTEGRATION RANGE: BECAUSE WE ARE LOOKING FOR THE PROPERTIES OF EC STATES, 
    # WE CAN SPEED UP THE SIMULATION BY SIMULATING THE GLV MODEL ONLY INSIDE A MU,SIGMA RECTANGLE INSIDE THE EC REGIME OF FIGURE 2A
    # (THIS PART OF CODE CAN BE INTERCHANGED TO INCLUDE THE WHOLE RANGE - MANY SIMULATIONS WILL THEN LEAD TO NO-EC AND WILL NOT BE STORED WHATSOEVER)
    
    #mu_1 = -np.random.uniform(meanAstart,meanAmax) 
    #mulist.append(mu_1)
    #varA= np.random.uniform(varAstart,varAmax)
    #sigmalist.append(varA)
    
    mu_1 = np.random.uniform(-1.5,-0.1) 
    mulist.append(mu_1)
    varA= np.random.uniform(0.05,0.8)
    sigmalist.append(varA)
    
    # FOR A GIVEN MU, SIGMA, FIND LARGEST POSSIBLE DIVERSITY
    
    mu = mu_1
    sigma = varA
    solution = find_integer_solution(-mu, sigma) # INVERT THE SIGN OF MU AS THE EQUATION IS WRITTEN IN THE -A_ij CONVENTION
   
    # FOR A GIVEN MU, SIGMA, FIND LARGEST POSSIBLE FRACTION OF EXCLUSIONARY ELEMENTS
   
    phi_val = stats.norm.cdf((-1 - mu) / sigma)
    ECf = 1 - (1 - phi_val) ** 2

    # FOR THIS GIVEN MU,SIGMA, BRING TOGETHER max S AND max fraction

    if solution in maxS:
        posicio = maxS.index(solution)
        if ECf > maxECpairs[posicio]:
             maxECpairs[posicio] = ECf
           
    # DEFINE A       
    A = np.random.normal(mu_1, varA, size=(S,S))
    
    # INCORPORATE RELATIVE SELF-REGULATION
    np.fill_diagonal(A,-np.ones(S))
    
    # SET COUNTERS   
    EC = 0
    BUC = 0
    SURV = []
    EC_pairs = []
    Number_exclusions_prime = []
    Fraction_exclusions_prime = []
    
    for j in range(reps):
        
        # INTEGRATE THE DYNAMICS
        def run(S, tmax=temps1, EPS=EPS,**kwargs):
            def eqs(t,x):
                dx = x*(1+np.dot(A,x))
                dx[x<=EPS]= 0 # SPECIES GOES EXTINCT BELOW THRESHOLD
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
                dx[x<=EPS]= 0
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
        
        # STATIONARY STATE WITH MORE THAN 2 SPECIES (REMEMBER WE ARE LOOKING ONLY AT EC STATES
                            
        if diff == 0 and np.sum(np.array(finalstate) > alive) > 2: 
            
            # FIND SURVIVING SPECIES TO GENERATE Aprime
            
            xzeros = [] # A BINARY STATE VECTOR
            for spp in range(S):
                if finalstate[spp]>alive: # if abundance is high enough, consider as survivor
                    xzeros.append(1)
                else:
                    xzeros.append(0) 
                       
            # Aprime CONTAINS ONLY THE INTERACTIONS BETWEEN SURVIVING SPECIES    
            
            extinct = np.where(np.array(xzeros) == 0)[0]
            Aprime = np.delete(np.delete(A,extinct,axis=0),extinct,axis=1)
            subset = Aprime.shape[0]
            
            
            # IS THERE EC? IF TRUE, MEASURE FRACTION OF PAIRS AND DIVERSITY
            if np.any(Aprime < -1): 
                
                EC_pairs.append(count_greater_than_one_pairs(Aprime))
                SURV.append(subset)
    
    surv_list.append(SURV) # X AXIS FIGURE 2C
    ecpairs_list.append(EC_pairs) # Y AXIS FIGURE 2C
         
    endclock = timer()
    print("Line runtime", endclock - startclock)




# NO MASK, THIS IS USED IN ESM TO FILTER E.G. STATES WITH GIVEN INTRANSITIVITY PROPERTIES - HERE IT HAS NO EFFECT.
surv_list = list(itertools.chain(*surv_list))
ecpairs_list = list(itertools.chain(*ecpairs_list))
surv_array = np.array(surv_list)
ecpairs_array = np.array(ecpairs_list)
mask = ecpairs_array < 1000 # CREATE AN UNREAL MASK WHERE NO ELEMENT IS ERASED
filtered_surv = surv_array[mask]
filtered_ecpairs = ecpairs_array[mask]

# DEFINE THE MINIMAL FRACTION: ITS ONLY 1 PAIR OVER ALL PAIRS = 1 / (S*(S-1)/2)
values = list(range(3, max(maxS)+1))
computed_values = [2 / (x * (x - 1)) for x in values]

# PLOT SIMULATIONS
scatter = plt.scatter(filtered_surv, filtered_ecpairs, color="purple", alpha=0.01)

# PLOT MIN ANALYTICAL FRACTION
plt.plot(values, computed_values, color="black", linestyle="--", label="2/(x*(x-1))")

# PLOT MAX ANALYTICAL FRACTION
plt.plot(maxS, maxECpairs, color="firebrick", linestyle="--")

# ZOOM IN RANGE FOR A NICE VISUALIZATION
plt.xlim(2.5, 13.5)

plt.xlabel('S')
plt.ylabel('Frac EC pairs')
plt.tight_layout()
nom = "Fraction_Excluding_Figure2C.png"
plt.savefig(nom, dpi=300)
plt.show()





























