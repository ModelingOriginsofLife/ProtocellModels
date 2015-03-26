using Distributions

function InitialPop(popsize,seqtypes,N)

    Population = Array(Vector{Int},popsize)
    equipartite=round(N/(seqtypes-1))
    equipartite=convert(Integer,equipartite)
    Founder=zeros(Integer,seqtypes-1)
    fill!(Founder,equipartite)
    push!(Founder,0)

    if sum(Founder)>N
    	Diff=N-sum(Founder)
    	where=rand(1:seqtypes-1)
        Founder[where]=Founder[where]+Diff
    elseif sum(Founder)<N
    	Diff=N-sum(Founder)
    	where=rand(1:seqtypes-1)
        Founder[where]=Founder[where]+Diff
    end

    for i=1:popsize
        Population[i]=Founder
    end

    return Population
end




### Test Parameters #####

    NumGens = 100
    popsize=100
    seqtypes=4
    N=200
    mu=0.01
    L=100