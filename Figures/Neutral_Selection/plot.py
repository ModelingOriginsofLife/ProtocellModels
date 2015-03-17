"""
plot.py
"""
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":

	df= pd.read_csv('fitness')



#Scatter of Fitness

	x=df['Time']
	y=df['Fitness']

# Get unique names

	uniq = list(set(df['Regime']))

# Set the color map

	z = range(1,len(uniq))
	hot = plt.get_cmap('Paired')
	cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

# Plot each regime

	for i in range(len(uniq)):
    	indx = df['Regime'] == uniq[i]
    	plt.scatter(x[indx], y[indx], s=15, color=scalarMap.to_rgba(i),label=uniq[i])

	plt.axis([0, 10000, 0, 1.1])
	plt.xlabel('Time')
	plt.ylabel('Average Fitness')
	plt.title('Effect of selection on different mutation regimes')
	plt.legend( loc='center right', numpoints = 1 )

#To save to a multipage PDF

	pp = PdfPages('figure_1.pdf')
	plt.savefig(pp,format='pdf')
	pp.close()
	