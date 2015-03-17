 #!/usr/bin/Rscript

#####################################################################
####      McFarland Calibration of the RTS instruments             ##
#####################################################################

#Input

options(echo=TRUE) # if you want see commands in output file
args <- commandArgs(trailingOnly = TRUE)
print(args)

# Process

name=args[1];

data=read.csv(name,header=FALSE);

data$V2=name;

total=nrow(data)-1;

data$V3=0:total

names(data)=c("Fitness","Regime","Time")


for (i in 2:length(args)){

 name=args[i];

 #Reading datasets

 dat2=read.csv(name,header=FALSE);
 dat2$V2=name;
 total=nrow(dat2)-1;
 dat2$V3=0:total;

 # Concatenating

 names(dat2)=c("Fitness","Regime","Time")


 data=rbind(data, dat2);


}

 #Writing 

 prefix="fitness";

 write.csv(data,prefix,sep="\t", row.names=FALSE, col.names=TRUE);
