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
    global test

    while (total != 2*N):
        sample=random.randrange(0,seqtypes)
        if sample == 0:
            protocell[0]=protocell[0]+1
            test = nprandom.binomial(1,q)
        elif test == 1:
            protocell[sample]=protocell[sample]+1
        else:
            protocell[0]=protocell[0]+1
        total=np.sum(protocell)

    return protocell

def ProtocellSplit(protocell,N):

    protocell=np.array(protocell,dtype=np.float)
    total=np.sum(protocell,dtype=np.float)
    Daughter_1=[0]*len(protocell)
    Daughter_2=[0]*len(protocell)

    global sum_daughter
    
    sum_daughter=sum(Daughter_1)
    
    while (sum_daughter != N):
        protocell=np.array(protocell,dtype=np.float)
        total=np.sum(protocell,dtype=np.float)
        protocell_freq=protocell/total
        values=np.arange(len(protocell))
        propia = sps.rv_discrete(name='propia', values=(values, protocell_freq))
        R = propia.rvs(size=1)
        R=R.tolist()
        R=int(R[0])

        while (protocell[R] == 0):

            R = propia.rvs(size=1)
            R=R.tolist()
            R=int(R[0])

        Daughter_1[R]=Daughter_1[R]+1
        sum_daughter=np.sum(Daughter_1)

        protocell[R]=protocell[R]-1

    Daughter_1=np.array(Daughter_1)
    Daughter_2=protocell

    return Daughter_1,Daughter_2

def InitialPop(popsize,seqtypes,N):

    Population = np.zeros((popsize,seqtypes))
    equipartite=N/(seqtypes-1)
    Founder=np.tile(equipartite,seqtypes)
    Founder[0]=0
    Reference=Founder[1:]
    MaxFit=0
    MaxFit=sps.gmean(Reference)

    if np.sum(Founder) < N:
        Diff=N-np.sum(Founder)
        where=random.randrange(1,seqtypes)
        Founder[where]=Founder[where]+Diff
    for i in range(popsize):
        Population[i]=Founder


    return Population,MaxFit

def Fitness(Population):

    fitness_vec=np.zeros(len(Population))

    for i in range(len(Population)):
        Compute=Population[i]
        No_omega=Compute[1:]
        fitness_vec[i]=sps.gmean(No_omega)
    return fitness_vec

def Measures(Population):
    fitness_vec=Fitness(Population)
    return fitness_vec 

def ForwardGen(Population,mu,L,N):

    "Pick the protocell"

    fitness_vec=Fitness(Population)
    total=np.sum(fitness_vec,dtype=np.float)
    fitness_freq=fitness_vec/total
    values=np.arange(len(Population))
    custm = sps.rv_discrete(name='custm', values=(values, fitness_freq))
    R = custm.rvs(size=1)
    R=R.tolist()
    R=int(R[0])

    "Moran Process"

    To_divide=SequenceDynSelf(Population[R],mu,L,N)
    Sons=ProtocellSplit(To_divide,N)

    sample=random.randrange(0,len(Population))

    Population[sample]=Sons[0]
    Population[R]=Sons[1]

    return Population


def SimPop(NumGens,popsize,seqtypes,N,mu,L):
    """
    Big Main
    """

    # binary flags

    plotreal = 0
    plotend = 1
    randenv = 0
    saveplot = 1
    showplot = 0

    #generate population with founder

    Population=InitialPop(popsize,seqtypes,N)[0]

    MaxFit=InitialPop(popsize,seqtypes,N)[1]

    #evolve population for NumGens storing mean population fitness

    AvgPopFit = np.zeros(NumGens)
  

    for i in range(NumGens):
        Population = ForwardGen(Population,mu,L,N)
        fitness_vec=Measures(Population)
        AvgPopFit[i] = np.mean(fitness_vec)/MaxFit

        # plot
    # ============================
    if plotend:
        # make directory for figure output
        if saveplot:
            figdir = "fig"
            if not os.path.exists(figdir):
                os.makedirs(figdir)

        #define font size
        plt.rc("font", size=20)
        plt.rc("text", usetex=True)

        # figure(1)
        fig1 = plt.figure(1)
        f1p1 = plt.plot(AvgPopFit,linestyle='none',marker='o')

        #To save to a multipage PDF
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(figdir + '/AvgPopFit.pdf')

        if saveplot:
            plt.savefig(pp,format='pdf')
            if showplot:
                plt.show()
        #plt.close()
        pp.close()

    return AvgPopFit, Population


if __name__ == "__main__":

	
    #import doctest
    #doctest.testmod()

    #read in command line parameters
    argv=sys.argv[1:]

    NumGens = int(argv[0])
    popsize=int(argv[1])
    seqtypes=int(argv[2])
    N=int(argv[3])
    mu=float(argv[4])
    L=int(argv[5])

    SimPop(NumGens,popsize,seqtypes,N,mu,L)


    # test

    NumGens = 10
    popsize=100
    seqtypes=4
    N=200
    mu=0.00001
    L=10
    test=0
    sum_daughter=0


protocell=[ 2,  7, 14, 12,  9, 11, 16, 11, 14, 11,  5,  6,  7,  8, 13, 10,  7,
   ...:        19,  9,  9]