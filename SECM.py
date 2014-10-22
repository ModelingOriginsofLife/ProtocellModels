"""
SECM.py
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nprandom
import scipy.stats as sps
import copy
import random as random

def SequenceDynSelf(protocell,mu,L,N):

	q = (1-mu)**L
	total=np.sum(protocell)

	while (total != 2*N):
		sample = random.sample(xrange(len(protocell)), 1)
		sample = np.array(sample)
		if sample == 0:
			protocell[0]=protocell[0]+1
		else:
			test = nprandom.binomial(1,q)
		if test == 1:
			protocell[sample]=protocell[sample]+1
		else:
			protocell[0]=protocell[0]+1
		total=np.sum(protocell)
	return protocell

def ProtocellSplit(protocell,N):
	
	protocell=np.array(protocell,dtype=np.float)
	total=np.sum(protocell,dtype=np.float)
	protocell_freq=protocell/total
	values=np.arange(len(protocell))
	custm = sps.rv_discrete(name='custm', values=(values, protocell_freq))
	R = custm.rvs(size=N)
	R=R.tolist()
	Daughter_1=[0]*len(protocell)
	Daughter_2=[0]*len(protocell)

	for i in range(0,len(protocell)):
		Daughter_1[i]=R.count(i)
	
	Daughter_1=np.array(Daughter_1)
	Daughter_2=protocell-Daughter_1

	return Daughter_1,Daughter_2

def InitialPop(popsize,seqtypes,N):

	Population = np.zeros((popsize,seqtypes))
	equipartite=N/seqtypes
	Founder=np.tile(equipartite,seqtypes)
	Founder[0]=0

	for i in range(popsize):
		Population[i]=Founder

	return Population

def ForwardGen(Population, GoalState, ConnectFlag = 1, MutateFlag = 1, ReproduceFlag=0, sigma=1, Term = 100, tau = 10, epsilon = 10**-4, a = 100, RateParam = 0.1, MagParam = 1, mu=0.1, b=0, MagParam1 = 1, MaintainFlag = 0):
    """
    # Script to propogate a population forward a generation given a set fitness GoalState
    # And strength of selection sigma. Individuals are randomly chosen to reporduce, and if their
    # Fitness is high enough, they succeed. Reproduction can be sexual (ReproduceFlag=1), asexual (ReproduceFlag=0)
    # Can involve mutation (MutateFlag), and changes in connectivity (ConnectFlag)
    """
    M = len(Population)
    PopFitness = np.zeros((M))
    N = len(Population[0,:,:])
    NewPop = np.zeros((M,N,N))
    for i in range(M):
        (ConFlag,PopState,Contime) = IterateInd(Population[i,:,:],GoalState,Term,tau,epsilon,a)
        if ConFlag == 1:
            PopFitness[i] = FitnessEval(PopState, GoalState, sigma)
        else:
            PopFitness[i] = 0

    PopNum = 0

    while PopNum < M:
        Z = nprandom.randint(0,M)
        if PopFitness[Z] > nprandom.random():

            if MutateFlag == 1:
                TempInd = MatMutate(Population[Z,:,:], RateParam, MagParam)

            if ConnectFlag == 1:
                TempInd = ConnectMutate(TempInd, mu, b, MagParam1, MaintainFlag)

            if ReproduceFlag == 1:
                P = nprandom.randint(0,M)
                if PopFitness[P] > nprandom.random():
                    TempInd = Reproduce(TempInd, Population[P,:,:])

            (ConFlagTemp, PopStateTemp, ConTimeTemp) = IterateInd(TempInd,GoalState,Term,tau,epsilon,a)

            # Cameron: should we really enforce this?
            # this prevents non-convergent individuals from
            # ever entering the population via mutation
            # I think it is enough that their fitness is 0
            if ConFlagTemp == 1:
                NewPop[PopNum,:,:] = copy.copy(TempInd)
                PopNum = PopNum + 1

    return NewPop, PopFitness

def SimPop(NumGens=10,NumInds=10,NumGenes=10):
    """
    simulate population of NumInds individuals each with NumGenes genes for
    NumGens generations

    input:  NumGens = number of generations
            NumInds = number of individuals
            NumGenes = number of genes per individual

    output: PopFitStore
    """

    #generate population with convergent founder
    ConFlag1=0
    while ConFlag1 != 1:
        Founder, Population = GeneratePop(NumInds,NumGenes)
        ConFlag1, FinState1, IntState1 = TestFounder(Founder)

    #evolve population for NumGens storing mean population fitness
    AvgPopFit = np.zeros(NumGens)
    NewPop = Population
    for i in range(NumGens):
        NewPop, PopFit = ForwardGen(NewPop,FinState1)
        AvgPopFit[i] = np.mean(PopFit)
    print AvgPopFit

    return Founder, Population, NewPop, AvgPopFit

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    #read in command line parameters
    argv=sys.argv[1:]

    if len(argv)==0:
        NumGens = 10
        NumInds = 10
        NumGenes = 10
    elif len(argv)==1:
        NumGens = int(argv[0])
        NumInds = 10
        NumGenes = 10
    elif len(argv)==2:
        NumGens = int(argv[0])
        NumInds = int(argv[1])
        NumGenes = 10
    elif len(argv)==3:
        NumGens = int(argv[0])
        NumInds = int(argv[1])
        NumGenes = int(argv[2])
    else:
        raise Exception("wrong number of input arguments: \npython wagbergmod.py NumGens NumInds NumGenes\n")

    SimPop(NumGens, NumInds, NumGenes)

