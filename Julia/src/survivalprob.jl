using Distributions

#-----------------------------
# Functions
#-----------------------------

function bin2dec(I::Array{Bool,1})
# bin2dec(map(Bool, dec2bin(5)))

  # convert boolean array to an integer
  return parse(Int,join(map(x->convert(Int,x),I)),2)

end

function dec2bin(N::Integer)

  # convert integer to boolean array
  map(x->parse(Int,x),split(bits(N),""))

end


function mutate(community::Array{Int64,1})
  newcommunity = zeros(community)
  for (indtype,indnum) in enumerate(community)
    # binindtype = map(Bool,dec2bin(indtype-1)[end-GL+1:end])
    binindtype = bits(indtype-1)[end-GL+1:end]
    for j=1:indnum
      tempindtype = copy(binindtype)
      # newindtype = bin2dec(mutate(tempindtype))
      tempindtype=mutate(tempindtype)
      # newcommunity[bin2dec(tempindtype)+1] += 1
      newcommunity[parse(Int,tempindtype,2)+1] += 1
    end
  end
  return newcommunity
end


function mutate(genome::ASCIIString)
  idx = mutating(GL,MF)
  newgenome = ""
  if !isempty(idx)
    for (i,bit) in enumerate(genome)
      if i in idx
        newgenome = string(newgenome,"0")
      else
        newgenome = string(newgenome,bit)
      end
    end
  else
    newgenome = genome
  end
  return newgenome
end

function mutate(genome::Array{Bool,1})
    # deleterious mutation
    idx = mutating(GL, MF)
    switchoff(idx, genome)

    # beneficial mutation
    # idx = mutating(GL,MF)
    # switchon(idx, c.genome)

    # if rand() < MF
    #     s = sign(rand() - 0.5)
    #     c.wantSporeSize += s
    # end
end

function mutating(n::Int64, f::Float64)
    find([x < y for (x,y) in zip(rand(n), f*ones(n))])
end

function switchoff(idx::Array{Int64}, gen::Array{Bool,1})
    if !isempty(idx)
        gen[idx] = false
    end
    gen
end

function switchon(idx::Array{Int64}, gen::Array{Bool,1})
    if !isempty(idx)
        gen[idx] = true
    end
    gen
end

function printarray(f,a)
  for i in a
    println(f,i)
  end
end

function printdebug(j,k,f,newpop,pop)
  if debugflag
    println(f,"")
    println(f,"--------------------------------------------")
    println(f,"")
    println(f,"generation number: $(j)")
    println(f,"community number: $(k)")
    println(f,"")

    println(f,"newpop:")
    printarray(f,newpop)
    println(f,"")

    println(f,"pop:")
    printarray(f,pop)
  end
end

function generation!(community,newpop,S,D)
  propagule = rand(Multinomial(M,community/N))
  newcommunity = rand(Multinomial(N,propagule/M))
  newcommunity = mutate(newcommunity)

  if bin2dec(map(Bool,sum(map(dec2bin,find(newcommunity)-1)))) == 2^GL - 1
      push!(newpop,newcommunity)
      S += 1
  else
      D += 1
  end

  return(S,D)
end



#----------------------------------
# main simulation
#----------------------------------

function main()

#-----------------------------
# Parameters
#-----------------------------

f = open("../log/survivalprob.log","w")

# Flags
global debugflag = false

# # Test Constants
# GL = 3   # number of potential genes in an individual
# N = 10   # number of individuals in a community
# P = 10    # number of communities

# MF = 0.01 # mutation rate
# M = 3    # number of individuals in a propagule
# G = 100   # number of generations
# T = 1  # number of trials

# GENUPDATE = 10 # number of generations between updates

# Run Constants
global GL = 10   # number of potential genes in an individual
global N = 100   # number of individuals in a community
global P = 100   # number of communities

global MF = 0.01 # mutation rate
global M = 15    # number of individuals in a propagule
global G = 5000  # number of generations
global T = 10   # number of trials

global GENUPDATE = 100 # number of generations between updates


survivalprobvec = zeros(T)

for i = 1:T

    # random population
    # pop = [rand(Multinomial(N,ones(2^GL)/(2^GL))) for i in 1:P]

    # optimal population
    pop = [[zeros(Int64,2^GL-1); N] for i in 1:P]

    if debugflag
      printarray(f,pop)
    end

    # community = rand(Multinomial(N,ones(2^GL)/(2^GL)))
    # propagule = rand(Multinomial(M,ones(2^GL)/(2^GL)))
    S = 0
    D = 0

    for j=1:G

        # Initialize survival and death parameters
        S = 0
        D = 0

        newpop = (Array{Int64,1})[]
        shuffle!(pop)

        for (k,community) in enumerate(pop)

          # trial 1: community-level reproduction
          S, D = generation!(community,newpop,S,D)

          if length(newpop)==P
            pop = copy(newpop)
            printdebug(j,k,f,newpop,pop)
            break
          end

          # trial 2: community-level reproduction
          S, D = generation!(community,newpop,S,D)

          if length(newpop)==P
            pop = copy(newpop)
            printdebug(j,k,f,newpop,pop)
            break
          end

          # print debug info to logfile
          printdebug(j,k,f,newpop,pop)

        end

        pop = copy(newpop)

        if debugflag
          println(f,"")
          println(f,"--------------------------------------------")
          println(f,"")
          println(f,"generation number: $(j)")
          println(f,"popsize: $(length(pop))")
        end

        if mod(j,GENUPDATE)==0

          println(
                    "generation: $(j)\t",
                    "popsize: $(length(pop))\t",
                    "rate: $(round(S/(S+D),3))\t",
                    "survivals: $(S)\t",
                    "deaths: $(D)\t",
                    "trial: $(i)"
                 )

        end

    end

    if length(pop) > 0
      survivalprobvec[i] = S/(S+D)
      println("survival rate: $(S/(S+D))")
      println("survivals: $(S)")
      println("deaths: $(D)")
    else
      survivalprobvec[i] = 0.0
    end

end

println(survivalprobvec)
println("mean survival: $(mean(survivalprobvec))")

println(f,survivalprobvec)
println(f,mean(survivalprobvec))

close(f)

end

# uncomment to run with
# julia probmod.jl
#
# main()


