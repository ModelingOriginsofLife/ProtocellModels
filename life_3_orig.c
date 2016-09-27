#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <float.h>
#include <string.h>

# define beta 1					// Inverse temperature factor

// Constants for random number generator (sourse: Numerical Recipes)

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

long seed;

// Random number generator

float ran2(long *idum)
{
	int j;
	long k;
	static long idum2=123456789;
	static long iy=0;
	static long iv[NTAB];
	float temp;
	
	if (*idum <= 0) {
		if (-(*idum) < 1) *idum=1;
		else *idum = -(*idum);
		idum2=(*idum);
		for (j=NTAB+7;j>=0;j--) {
			k=(*idum)/IQ1;
			*idum=IA1*(*idum-k*IQ1)-k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy=iv[0];
	}
	k=(*idum)/IQ1;
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	if (*idum < 0) *idum += IM1;
	k=idum2/IQ2;
	idum2=IA2*(idum2-k*IQ2)-k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j=iy/NDIV;
	iy=iv[j]-idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}

// With this function we define the potential energies of all the different possible pairwise interactions.
// Therefore we are constructing an NxNx7 matrix where N is the number of distinct particle types.
// There are 7 layers to this matrix becuase there are 7 different pairs of relative positions (hexagonal grid)
// Membrane particles can exist in one of six orientations, each separated by an angle of pi/6 (they are double-
// ended particles (like two lipids stuck end to end by their tails)
void init_phi(double *f,int dep,double ***phi_all){
	int i,j,k;
	
	// Membrane-membrane interactions at site 0
	phi_all[0][0][0]=f[2];
	phi_all[0][1][0]=f[2]+f[3]/3;
	phi_all[0][2][0]=f[2]+2*f[3]/3;
	phi_all[0][3][0]=f[2]+f[3];
	phi_all[0][4][0]=f[2]+2*f[3]/3;
	phi_all[0][5][0]=f[2]+f[3]/3;
	for(i=1;i<6;i++){
		for(j=0;j<6;j++){
			phi_all[i][j][0]=phi_all[i-1][(j+5)%6][0];
		}
	}
	
	// Neutral interactions at site 0
	for(i=6;i<12;i++){
		for(j=0;j<dep;j++){
			phi_all[i][j][0]=f[1];
			phi_all[j][i][0]=f[1];
		}
	}
	
	// Hydrophilic interactions at site 0
	for(i=12;i<dep;i++){
		for(j=0;j<dep;j++){
			if(j<6){							// Hydrophilic-membrane interactions
				phi_all[i][j][0]=20.0*f[0];
				phi_all[j][i][0]=20.0*f[0];
			}
			else if(j>11){						// Hydrophilic-hydrophilic interactions
				phi_all[i][j][0]=f[0];
			}
		}
	}
	
	// Fill in the remaining potentials for interactions over a distance of 1 lattice spacing
	// Most interactions are just reduced by a factor of 3. Those that aren't simply re-scaled
	// will be corrected below
	for(i=0;i<dep;i++){
		for(j=0;j<dep;j++){
			for(k=1;k<7;k++){
				phi_all[i][j][k]=phi_all[i][j][0]/3;
			}
		}
	}
	
	// Now we correct the anisotropic interactions
	// Hydrophobic-neutral interactions for separated particles
	phi_all[6][0][1]=4*f[1]/5;
	phi_all[6][1][1]=f[1]/2;
	phi_all[6][2][1]=f[1]/10;
	phi_all[6][3][1]=0.0;
	phi_all[6][4][1]=f[1]/10;
	phi_all[6][5][1]=f[1]/2;
	
	// Copy for the other neutral particles
	for(i=6;i<12;i++){
		for(j=0;j<6;j++){
			phi_all[i][j][1]=phi_all[6][j][1];
			phi_all[j][i][1]=phi_all[i][j][1];
		}
	}
	
	// Hydrophobic-hydrophilic interactions at site 1
	phi_all[12][0][1]=16*f[0];
	phi_all[12][1][1]=10*f[0];
	phi_all[12][2][1]=2*f[0];
	phi_all[12][3][1]=0;
	phi_all[12][4][1]=2*f[0];
	phi_all[12][5][1]=10*f[0];
	
	// Copy for the other hydrophilic particles
	for(i=12;i<dep;i++){
		for(j=0;j<6;j++){
			phi_all[i][j][1]=phi_all[12][j][1];
			phi_all[j][i][1]=phi_all[i][j][1];
		}
	}
	
	// Now we make use of the lattice symmetry to fill in the interactions for the other
	// neighbour sites for the neutral and hydrophilic particle interactions
	for(k=2;k<7;k++){
		for(i=6;i<dep;i++){
			for(j=0;j<6;j++){
				phi_all[i][j][k]=phi_all[i][(j+4)%6][k-1];
				phi_all[j][i][k]=phi_all[(j+4)%6][i][k-1];
			}
		}
	}
}

// Initialise grid using values for initial concentrations i_c
void init_grid(int div,int x_s,int dep,double *i_c,int ***n,double ***PHI){
	int i,j,k,c;
	// First reset everything to 0
	for(i=0;i<div+2;i++){
		for(j=0;j<x_s+2;j++){
			for(k=0;k<dep;k++){
				n[i][j][k]=0;
				PHI[i][j][k]=0.0;
			}
		}
	}
	// Now randomly place particles on grid according to initial concentrations
	for(k=0;k<dep;k++){
		for(c=0;c<floor(i_c[k]*div*x_s+0.5);c++){
			i=floor(div*ran2(&seed))+1;
			j=floor(x_s*ran2(&seed))+1;
			n[i][j][k]++;
		}
	}
}

// This function simply totals up all the particles in the whole grid
int sum_all_procs(int div,int x_s,int dep,int ***n){
	int i,j,k,p_sum=0,all_sum;
	for(i=1;i<div+1;i++){
		for(j=1;j<x_s+1;j++){
			for(k=0;k<dep;k++){
				p_sum+=n[i][j][k];
			}
		}
	}
	MPI_Allreduce(&p_sum,&all_sum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	return all_sum;
}

// This function simply prints out the numbers of particles at each grid cell summed over the
// types from st_row to end_row
void add_pr(int y_s,int x_s,int ***n_final,int st_row,int end_row){
	int i,j,k,t_sum;
	for(i=0;i<y_s;i++){
		for(j=1;j<x_s+1;j++){
			t_sum=0;
			for(k=st_row;k<end_row+1;k++){
				t_sum+=n_final[i][j][k];
			}
			printf("%2d ",t_sum);
		}
		printf("\n");
	}
	printf("\n");
}

// This function just implements periodic boundary conditions in the horizontal direction
void per_BCs_hor_int(int div,int x_s,int dep,int ***n){
	int i;
	for(i=1;i<div+1;i++){
		memcpy(&n[i][0][0],&n[i][x_s][0],dep*sizeof(int));
		memcpy(&n[i][x_s+1][0],&n[i][1][0],dep*sizeof(int));
	}
}

void calc_PHI(int div,int x_s,int dep,int **coords,int **coords_2,double ***phi_all,int ***n,double ***PHI){
	int t_1,t_2,nbr,i,j;
	
	for(i=1;i<div/2+1;i++){
		for(j=1;j<x_s+1;j++){
			// Now calculate new potential energy values
			for(t_1=0;t_1<dep;t_1++){
				// First reset PHI
				PHI[i*2][j][t_1]=0.0;
				PHI[i*2-1][j][t_1]=0.0;
				// Now sum total potential contributions from all neighbours and all other particle types
				for(t_2=0;t_2<dep;t_2++){
					for(nbr=0;nbr<7;nbr++){
						PHI[i*2][j][t_1]+=phi_all[t_1][t_2][nbr]*
						n[i*2+coords[nbr][0]][j+coords[nbr][1]][t_2];
						PHI[i*2-1][j][t_1]+=phi_all[t_1][t_2][nbr]*
						n[i*2-1+coords_2[nbr][0]][j+coords_2[nbr][1]][t_2];
					}
				}
				// Now we correct for same site interactions between particle of the same type
				// In that case the value of n must be reduced by 1 since particles don't repel themselves
				PHI[i*2][j][t_1]+=phi_all[t_1][t_1][0]*
				(n[i*2+coords[0][0]][j+coords[0][1]][t_1]-1)*
				(n[i*2+coords[0][0]][j+coords[0][1]][t_1]>0)-
				phi_all[t_1][t_1][0]*n[i*2+coords[0][0]][j+coords[0][1]][t_1];
				
				PHI[i*2-1][j][t_1]+=phi_all[t_1][t_1][0]*
				(n[i*2-1+coords_2[0][0]][j+coords_2[0][1]][t_1]-1)*
				(n[i*2-1+coords_2[0][0]][j+coords_2[0][1]][t_1]>0)-
				phi_all[t_1][t_1][0]*n[i*2-1+coords_2[0][0]][j+coords_2[0][1]][t_1];
			}
		}
	}
}

// This function just implements periodic boundary conditions in the horizontal direction
void per_BCs_hor_db(int div,int x_s,int dep,double ***grid){
	int i;
	for(i=1;i<div+1;i++){
		memcpy(&grid[i][0][0],&grid[i][x_s][0],dep*sizeof(double));
		memcpy(&grid[i][x_s+1][0],&grid[i][1][0],dep*sizeof(double));
	}
}

// This function makes a copy of the main grid n upon n_2. It also initialises
// the boundary rows and columns of n_2 to 0 since particles will need to exclusively 
// migrate to these regions
void copy_n(int div,int x_s,int dep,int **z_s,int d_len,int d_len_2,int ***n,int ***n_2){
	int i;
	// First make a copy of n
	memcpy(&n_2[0][0][0],&n[0][0][0],d_len_2*sizeof(int));
	// Now set upper and lower rows to zero
	memcpy(&n_2[0][0][0],&z_s[0][0],d_len*sizeof(int));
	memcpy(&n_2[div+1][0][0],&z_s[0][0],d_len*sizeof(int));
	// Now set first and last columns to zero
	for(i=1;i<div+1;i++){
		memcpy(&n_2[i][0][0],&z_s[0][0],dep*sizeof(int));
		memcpy(&n_2[i][x_s+1][0],&z_s[0][0],dep*sizeof(int));
	}
}

void update_all(int odd,int height,int width,int num_As,int num_Bs,int dep,
				double rot,double mu,double mu_B,double k_m,double k_a,double k_y,
				double k_b,double s_a,double s_m,double s_b,double C_a,double C_m,
				double C_b,double C_y,double *del_H,int **coords,double *k_dif,
				double ***PHI,int ***n,int ***n_2){
	
	int i,j,k,c,no_m,no_r,n_loc,new_t,p_ind;
	double x,p_sum,p_1,k_x_m,k_m_y,k_sum,del_E,n_pr;
	
	for(i=1;i<height/2+1;i++){
		for(j=1;j<width+1;j++){
			for(k=0;k<dep;k++){
				if(n[i*2-odd][j][k]>0){
					for(c=0;c<n[i*2-odd][j][k];c++){
						// Calculate whether particle will move.
						no_m=1;
						x=ran2(&seed);
						p_sum=0.0;
						for(n_loc=1;n_loc<7;n_loc++){
							del_E=PHI[i*2-odd+coords[n_loc][0]][j+coords[n_loc][1]][k]-PHI[i*2-odd][j][k];
							p_sum+=k_dif[k]*del_E/(exp(beta*del_E)-1);
							if(p_sum>x){
								// Particle moves to position n_loc
								no_m=0;
								break;
							}
						}
						// Calculate whether the particle will rotate or react
						no_r=1;
						if(k<6){
							// Membrane particle
							x=ran2(&seed);
							p_sum=0.0;
							for(new_t=0;new_t<6;new_t++){
								if(new_t!=k){	  
									del_E=PHI[i*2-odd][j][new_t]-PHI[i*2-odd][j][k];
									p_sum+=rot*del_E/(exp(beta*del_E)-1);
									if(p_sum>x){
										// Particle rotates (changes to type new_t)
										no_r=0;
										break;
									}
								}
							}
							// Now check for reaction. We must calculate rate constant k_x_m
							x=ran2(&seed);
							k_x_m=k_m;
							for(p_ind=14;p_ind<13+num_As;p_ind++){
								k_x_m+=C_m*n[i*2-odd][j][p_ind]*pow(2,p_ind-14);
							}
							del_E=del_H[0]-del_H[3]+PHI[i*2-odd][j][6]-PHI[i*2-odd][j][k];
							p_1=k_x_m*del_E/(exp(beta*del_E)-1);
							if(p_1>x){
								// Turn back into resource particle
								no_r=0;
								new_t=6;
							}
							else{
								del_E=del_H[1]-del_H[3]+PHI[i*2-odd][j][7]-PHI[i*2-odd][j][k];
								k_m_y=k_y;
								for(p_ind=14+num_As;p_ind<dep;p_ind++){
									k_m_y+=C_y*n[i*2-odd][j][p_ind]*pow(2,p_ind-(14+num_As));
								}
								if((p_1+k_m_y*del_E/(exp(beta*del_E)-1))>x){
									// Turn into waste particle
									no_r=0;
									new_t=7;
								}
							}
						}
						else if(k==6){
							// Membrane resource particle. We must calculate rate constant k_x_m
							k_x_m=k_m;
							for(p_ind=14;p_ind<13+num_As;p_ind++){
								k_x_m+=C_m*n[i*2-odd][j][p_ind]*pow(2,p_ind-14);
							}
							x=ran2(&seed);
							p_sum=0.0;
							for(new_t=0;new_t<6;new_t++){
								del_E=del_H[3]-del_H[0]+PHI[i*2-odd][j][new_t]-PHI[i*2-odd][j][k];
								p_sum+=k_x_m*del_E/(exp(beta*del_E)-1);
								if(p_sum>x){
									// Turn into membrane particle new_t
									no_r=0;
									break;
								}
							}
							if(no_r==1){
								del_E=del_H[1]-del_H[0]+PHI[i*2-odd][j][7]-PHI[i*2-odd][j][k];
								if((p_sum+k_y*del_E/(exp(beta*del_E)-1))>x){
									// Turn into membrane waste particle
									no_r=0;
									new_t=7;
								}
							}
						}
						else if(k==7){
							// Membrane waste particle
							x=ran2(&seed);
							p_sum=0.0;
							// Calculate k_m_y
							k_m_y=k_y;
							for(p_ind=13+num_As;p_ind<dep;p_ind++){
								k_m_y+=C_y*n[i*2-odd][j][p_ind]*pow(2,p_ind-(13+num_As));
							}
							for(new_t=0;new_t<6;new_t++){
								del_E=del_H[3]-del_H[1]+PHI[i*2-odd][j][new_t]-PHI[i*2-odd][j][k];
								p_sum+=k_m_y*del_E/(exp(beta*del_E)-1);
								if(p_sum>x){
									// Turn into membrane particle new_t
									no_r=0;
									break;
								}
							}
							if(no_r==1){
								del_E=del_H[0]-del_H[1]+PHI[i*2-odd][j][6]-PHI[i*2-odd][j][k];
								if((p_sum+(k_y+s_m)*del_E/(exp(beta*del_E)-1))>x){
									// Turn into membrane resource particle
									no_r=0;
									new_t=6;
								}
							}
						}
						else if(k==8){
							// Catalyst resource particle. We must calculate k_x_a
							x=ran2(&seed);
							k_sum=0.0;
							for(p_ind=13;p_ind<13+num_As;p_ind++){
								k_sum+=n[i*2-odd][j][p_ind];
							}
							p_sum=0.0;
							for(new_t=13;new_t<13+num_As;new_t++){
								del_E=del_H[2]-del_H[0]+PHI[i*2-odd][j][new_t]-PHI[i*2-odd][j][k];
								if(new_t==13){
									n_pr=(1-mu)*n[i*2-odd][j][new_t]+mu*n[i*2-odd][j][new_t+1];
								}
								else if(new_t==12+num_As){
									n_pr=(1-mu)*n[i*2-odd][j][new_t]+mu*n[i*2-odd][j][new_t-1];
								}
								else{
									n_pr=(1-2*mu)*n[i*2-odd][j][new_t]+mu*n[i*2-odd][j][new_t+1]+mu*n[i*2-odd][j][new_t-1];
								}
								p_sum+=(k_a+n_pr*C_a*k_sum)*del_E/(exp(beta*del_E)-1);
								if(p_sum>x){
									// Turn into catalyst particle new_t
									no_r=0;
									break;
								}
							}
							if(no_r==1){
								del_E=del_H[1]-del_H[0]+PHI[i*2-odd][j][9]-PHI[i*2-odd][j][k];
								if((p_sum+k_y*del_E/(exp(beta*del_E)-1))>x){
									// Turn into catalyst waste particle
									no_r=0;
									new_t=9;
								}
							}
						}
						else if(k==9){
							// Catalyst waste particle
							x=ran2(&seed);
							p_sum=0.0;
							for(new_t=13;new_t<13+num_As;new_t++){
								del_E=del_H[2]-del_H[1]+PHI[i*2-odd][j][new_t]-PHI[i*2-odd][j][k];
								p_sum+=k_y*del_E/(exp(beta*del_E)-1);
								if(p_sum>x){
									// Turn into catalyst particle new_t
									no_r=0;
									break;
								}
							}
							if(no_r==1){
								del_E=del_H[0]-del_H[1]+PHI[i*2-odd][j][8]-PHI[i*2-odd][j][k];
								if((p_sum+(k_y+s_a)*del_E/(exp(beta*del_E)-1))>x){
									// Turn into catalyst resource particle
									new_t=8;
									no_r=0;
								}
							}
						}
						else if(k==10){
							// Catalyst B resource particle
							x=ran2(&seed);
							k_sum=0.0;
							for(p_ind=13+num_As;p_ind<dep;p_ind++){
								k_sum+=n[i*2-odd][j][p_ind];
							}
							p_sum=0.0;
							for(new_t=13+num_As;new_t<dep;new_t++){
								del_E=del_H[2]-del_H[0]+PHI[i*2-odd][j][new_t]-PHI[i*2-odd][j][k];
								if(new_t==13+num_As){
									n_pr=(1-mu_B)*n[i*2-odd][j][new_t]+mu_B*n[i*2-odd][j][new_t+1];
								}
								else if(new_t==12+num_As+num_Bs){
									n_pr=(1-mu_B)*n[i*2-odd][j][new_t]+mu_B*n[i*2-odd][j][new_t-1];
								}
								else{
									n_pr=(1-2*mu_B)*n[i*2-odd][j][new_t]+mu_B*n[i*2-odd][j][new_t+1]+mu_B*n[i*2-odd][j][new_t-1];
								}
								p_sum+=(k_b+n_pr*C_b*k_sum)*del_E/(exp(beta*del_E)-1);
								if(p_sum>x){
									// Turn into catalyst B particle new_t
									no_r=0;
									break;
								}
							}
							if(no_r==1){
								del_E=del_H[1]-del_H[0]+PHI[i*2-odd][j][11]-PHI[i*2-odd][j][k];
								if((p_sum+k_y*del_E/(exp(beta*del_E)-1))>x){
									// Turn into catalyst B waste particle
									no_r=0;
									new_t=11;
								}
							}
						}
						else if(k==11){
							// Catalyst B waste particle
							x=ran2(&seed);
							p_sum=0.0;
							for(new_t=13+num_As;new_t<dep;new_t++){
								del_E=del_H[2]-del_H[1]+PHI[i*2-odd][j][new_t]-PHI[i*2-odd][j][k];
								p_sum+=k_y*del_E/(exp(beta*del_E)-1);
								if(p_sum>x){
									// Turn into catalyst B particle new_t
									no_r=0;
									break;
								}
							}
							if(no_r==1){
								del_E=del_H[0]-del_H[1]+PHI[i*2-odd][j][10]-PHI[i*2-odd][j][k];
								if((p_sum+(k_y+s_b)*del_E/(exp(beta*del_E)-1))>x){
									// Turn into catalyst B resource particle
									new_t=10;
									no_r=0;
								}
							}
						}
						else if(k>12 && k<13+num_As){
							// Catalyst particle. We must calculate k_x_a
							x=ran2(&seed);
							k_sum=0.0;
							for(p_ind=13;p_ind<13+num_As;p_ind++){
								k_sum+=n[i*2-odd][j][p_ind];
							}
							if(k==13){
								n_pr=(1-mu)*n[i*2-odd][j][k]+mu*n[i*2-odd][j][k+1];
							}
							else if(k==12+num_As){
								n_pr=(1-mu)*n[i*2-odd][j][k]+mu*n[i*2-odd][j][k-1];
							}
							else{
								n_pr=(1-2*mu)*n[i*2-odd][j][k]+mu*n[i*2-odd][j][k+1]+mu*n[i*2-odd][j][k-1];;
							}
							k_sum*=C_a*n_pr;
							k_sum+=k_a;
							del_E=del_H[0]-del_H[2]+PHI[i*2-odd][j][8]-PHI[i*2-odd][j][k];
							p_1=k_sum*del_E/(exp(beta*del_E)-1);
							if(p_1>x){
								// Turn into catalyst resource particle
								new_t=8;
								no_r=0;
							}
							else{
								del_E=del_H[1]-del_H[2]+PHI[i*2-odd][j][9]-PHI[i*2-odd][j][k];
								if((p_1+k_y*del_E/(exp(beta*del_E)-1))>x){
									// Turn into catalyst waste particle
									new_t=9;
									no_r=0;
								}
							}
						}
						else if(k>12+num_As){
							// Catalyst B particle. We must calculate k_x_b
							x=ran2(&seed);
							k_sum=0.0;
							for(p_ind=13+num_As;p_ind<dep;p_ind++){
								k_sum+=n[i*2-odd][j][p_ind];
							}
							if(k==13+num_As){
								n_pr=(1-mu_B)*n[i*2-odd][j][k]+mu_B*n[i*2-odd][j][k+1];
							}
							else if(k==12+num_As+num_Bs){
								n_pr=(1-mu_B)*n[i*2-odd][j][k]+mu_B*n[i*2-odd][j][k-1];
							}
							else{
								n_pr=(1-2*mu_B)*n[i*2-odd][j][k]+mu_B*n[i*2-odd][j][k+1]+mu_B*n[i*2-odd][j][k-1];;
							}
							k_sum*=C_b*n_pr;
							k_sum+=k_b;
							del_E=del_H[0]-del_H[2]+PHI[i*2-odd][j][10]-PHI[i*2-odd][j][k];
							p_1=k_sum*del_E/(exp(beta*del_E)-1);
							if(p_1>x){
								// Turn into catalyst B resource particle
								new_t=10;
								no_r=0;
							}
							else{
								del_E=del_H[1]-del_H[2]+PHI[i*2-odd][j][11]-PHI[i*2-odd][j][k];
								if((p_1+k_y*del_E/(exp(beta*del_E)-1))>x){
									// Turn into catalyst B waste particle
									new_t=11;
									no_r=0;
								}
							}
						}
						// Now we know whether the particle will move, rotate or react,
						// or a combination of them
						if(no_m==0){
							// The particle will move from its current position to the neighbouring
							// position n_loc. Remove from current position
							n_2[i*2-odd][j][k]--;
							if(no_r==0){
								// The particle is also reacting or rotating
								n_2[i*2-odd+coords[n_loc][0]][j+coords[n_loc][1]][new_t]++;
							}
							else{
								// The particle is just moving
								n_2[i*2-odd+coords[n_loc][0]][j+coords[n_loc][1]][k]++;
							}
						}
						else if(no_r==0){
							// The particle is not moving, but it is reacting or rotating.
							// Remove from current list.
							n_2[i*2-odd][j][k]--;
							n_2[i*2-odd][j][new_t]++;
						}
					}
				}
			}
		}
	}
}

void add_strip(int x_s,int row,int dep,int **n_strip,int ***n_2){
	int j,k;
	for(j=0;j<x_s+2;j++){
		for(k=0;k<dep;k++){
			n_2[row][j][k]+=n_strip[j][k];
		}
	}
}

void fill_Bs(int div,int x_s,int dep,int ***grid){
	int i,k;
	for(i=1;i<div+1;i++){
		for(k=0;k<dep;k++){
			grid[i][1][k]+=grid[i][x_s+1][k];
			grid[i][x_s][k]+=grid[i][0][k];
		}
	}
}

int main(int argc, char **argv){
	
	// Define constants
	int i,j,k,l;
	int num_As=10,num_Bs=10,dep=13+num_As+num_Bs;	// Total number of possible catalyst species
	int t,t_end=2e6,t_int=t_end/500;				// Time variables
	int num_procs,rank,d_len,d_len_2,d_len_3,above,below;	// MPI variables
	int tot;										// Total particle count
	int x_s=320,y_s=x_s,div;							// Grid size
	int ***n,***n_2,**n_strip,***n_final;			// Pointers for grids
	int *space_n,*space_n_2,*space_n_final;			// Memory block pointers
	int **coords,**coords_2;						// Neighbourhood coordinates
	int **z_s;										// Array of zeroes
	double rot=0.01;								// Membrane particle rotation coefficient
	double mu=5e-9;									// A-catalyst mutation factor
	double mu_B=5e-9;								// B-catalyst mutation factor
	double k_m=5e-9;								// Membrane synthesis base rate
	double k_a=5e-9;								// A-catalyst synthesis base rate
	double k_y=5e-5;								// Membrane decay base rate
	double k_b=5e-9;								// B-catalyst synthesis base rate
	double s_a=8.0;									// A-catalyst waste particle recycle factor
	double s_m=10.0;								// Membrane particle waste recycle factor
	double s_b=8.0;									// B-catalyst waste particle recycle factor
	double C_a=5e-6;								// A-catalyst synthesis mutant increase factor
	double C_m=1e-6;								// Membrane synthesis mutant increase factor
	double C_b=5e-6;								// B-catalyst synthesis mutant increase factor
	double C_y=1e-6;								// Membrane decay mutant increase factor
	double ***phi_all,***PHI;						// Pointers for potential grids
	double *space_PHI,*space_phi;					// Memory blocks for potential grids
	double k_dif[dep];								// Diffusion coefficients
	double del_H[4]={12.0,0.0,6.0,4.0};				// Enthalpy change values
	double f[4];									// Interaction constants
	double init_concs[dep];							// Initial concentration vector
	
	// Initialise MPI
	MPI_Status status;
	MPI_Comm Comm_cart;
	int dims[1],periods[1];
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	dims[0]=num_procs;
	periods[0]=1;
	MPI_Cart_create(MPI_COMM_WORLD,1,dims,periods,0,&Comm_cart);
	MPI_Cart_shift(Comm_cart,0,1,&above,&below);
	
	// Seed random number generator using process rank
	//seed=-(signed)time(NULL)*rank*rank;
	seed=rank*(3+rank*5);
	
	// Calculate labour division.
	div=y_s/num_procs;
	
	// Declare main and secondary grids and potential grid PHI
	n=malloc((div+2)*sizeof(int **));
	n_2=malloc((div+2)*sizeof(int **));
	PHI=malloc((div+2)*sizeof(double **));
	// Now we actually allocate the blocks of memory before setting up all the pointers for the three dimensions
	space_n=malloc((div+2)*(x_s+2)*dep*sizeof(int));
	space_n_2=malloc((div+2)*(x_s+2)*dep*sizeof(int));
	space_PHI=malloc((div+2)*(x_s+2)*dep*sizeof(double));
	// Now assign all the pointers
	for(i=0;i<div+2;i++){
		n[i]=malloc((x_s+2)*sizeof(int *));
		n_2[i]=malloc((x_s+2)*sizeof(int *));
		PHI[i]=malloc((x_s+2)*sizeof(double *));
		for(j=0;j<x_s+2;j++){
			n[i][j]=space_n+i*(x_s+2)*dep+j*dep;
			n_2[i][j]=space_n_2+i*(x_s+2)*dep+j*dep;
			PHI[i][j]=space_PHI+i*(x_s+2)*dep+j*dep;
		}
	}
	
	// Now do the same for the complete grid
	n_final=malloc(y_s*sizeof(int **));
	space_n_final=malloc(y_s*(x_s+2)*dep*sizeof(int));
	for(i=0;i<y_s;i++){
		n_final[i]=malloc((x_s+2)*sizeof(int *));
		for(j=0;j<x_s+2;j++){
			n_final[i][j]=space_n_final+i*(x_s+2)*dep+j*dep;
		}
	}
	// And for the exchange strip
	n_strip=malloc((x_s+2)*sizeof(int *));
	n_strip[0]=malloc((x_s+2)*dep*sizeof(int));
	z_s=malloc((x_s+2)*sizeof(int *));
	z_s[0]=malloc((x_s+2)*dep*sizeof(int));
	for(j=0;j<x_s+2;j++){
		n_strip[j]=n_strip[0]+j*dep;
		z_s[j]=z_s[0]+j*dep;
	}
	// And finally for the interaction strengths array
	phi_all=malloc(dep*sizeof(double **));
	space_phi=malloc(dep*dep*7*sizeof(double));
	for(i=0;i<dep;i++){
		phi_all[i]=malloc(dep*sizeof(double *));
		for(j=0;j<dep;j++){
			phi_all[i][j]=space_phi+i*dep*7+j*7;
		}
	}
	
	// Define the two sets of neighbour coordinates
	coords=malloc(7*sizeof(int *));
	coords_2=malloc(7*sizeof(int *));
	coords[0]=malloc(7*2*sizeof(int));
	coords_2[0]=malloc(7*2*sizeof(int));
	for(i=0;i<7;i++){
		coords[i]=coords[0]+2*i;
		coords_2[i]=coords_2[0]+2*i;
	}
	
	coords[0][0]=0; coords[0][1]=0; coords[1][0]=0; coords[1][1]=1;
	coords[2][0]=-1; coords[2][1]=0; coords[3][0]=-1; coords[3][1]=-1;
	coords[4][0]=0; coords[4][1]=-1; coords[5][0]=1; coords[5][1]=-1;
	coords[6][0]=1; coords[6][1]=0;
	coords_2[0][0]=0; coords_2[0][1]=0; coords_2[1][0]=0; coords_2[1][1]=1;
	coords_2[2][0]=-1; coords_2[2][1]=1; coords_2[3][0]=-1; coords_2[3][1]=0;
	coords_2[4][0]=0; coords_2[4][1]=-1; coords_2[5][0]=1; coords_2[5][1]=0;
	coords_2[6][0]=1; coords_2[6][1]=1;
	
	// Define zeros row
	for(j=0;j<x_s+2;j++){
		for(k=0;k<dep;k++){
			z_s[j][k]=0;
		}
	}
	
	// Define individual repulsions
	f[0]=0.01;		// Interactions between hydrophilic and all other particles. It's 
					// magnitude is increased for interactions with hydrophobic particles
	f[1]=0.001;		// Interactions between neutral and all other particles
	f[2]=0.01;		// Interactions between membrane particles and themselves (self-repulsion)
	f[3]=0.2;		// Extra repulsion due to mis-alignment between membrane particles
	
	// Define repulsive interactions.
	init_phi(f,dep,phi_all);
	
	// Define diffusion constants
	for(i=0;i<dep;i++){
		if(i<13 && i>5){
			k_dif[i]=0.01;		// Neutral and water particles
		}
		else{
			k_dif[i]=0.003;		// Membrane and catalyst particles
		}
	}
	
	// Initialise concentrations
	// Membrane particles
	for(i=0;i<6;i++){
		init_concs[i]=0.0;
	}
	init_concs[6]=5.0;		// Membrane resource particles X_m
	init_concs[7]=5.0;		// Membrane waste particles Y_m
	init_concs[8]=2.5;		// Catalyst resource particles X_A
	init_concs[9]=2.5;		// Catalyst waste particles	Y_A
	init_concs[10]=2.5;		// Catalyst B resource particles X_B
	init_concs[11]=2.5;		// Catalyst B waste particles Y_B
	init_concs[12]=10.0;	// Water
	
	// Catalyst A particles
	init_concs[13]=5.0;
	for(i=14;i<13+num_As;i++){
		init_concs[i]=0.0;
	}
	
	// Catalyst B particles
	init_concs[13+num_As]=5.0;
	for(i=14+num_As;i<dep;i++){
		init_concs[i]=0.0;
	}
	
	// Initialise grid
	init_grid(div,x_s,dep,init_concs,n,PHI);	
	
	d_len=(x_s+2)*dep;
	d_len_2=(div+2)*(x_s+2)*dep;
	d_len_3=div*(x_s+2)*dep;
	
	MPI_Gather(&n[1][0][0],d_len_3,MPI_INT,&n_final[0][0][0],d_len_3,MPI_INT,0,MPI_COMM_WORLD);
	
	// Check the initial total number of particles
	tot=sum_all_procs(div,x_s,dep,n);
	
	// Print out starting grid
	if(rank==0){
		printf("\n%d %d %d %d %d\n\n",y_s,x_s,t_end,t_int,tot);
		add_pr(y_s,x_s,n_final,0,5);							// Membrane particles
		//add_pr(y_s,x_s,n_final,6,8);							// Resource particles
		//add_pr(y_s,x_s,n_final,9,11);							// Waste particles
		add_pr(y_s,x_s,n_final,13,12+num_As);					// A-catalysts
		add_pr(y_s,x_s,n_final,13+num_As,12+num_As+num_Bs);		// B-catalysts
		add_pr(y_s,x_s,n_final,12,12);							// Water
	}
	
	// Run simulation
	for(t=1;t<=t_end;t++){
		
		// Now we need to pad all the grids using the periodic boundary conditions
		// First in the horizontal direction
		per_BCs_hor_int(div,x_s,dep,n);
		
		// Exchange boundary rows between processes. Send boundary data to process 
		// below and receive from process above
		MPI_Sendrecv(&n[div][0][0],d_len,MPI_INT,below,3,&n[0][0][0],d_len,MPI_INT,above,3,MPI_COMM_WORLD,&status);
		
		// Now send and receive in the opposite direction
		MPI_Sendrecv(&n[1][0][0],d_len,MPI_INT,above,4,&n[div+1][0][0],d_len,MPI_INT,below,4,MPI_COMM_WORLD,&status);
		
		// Calculate potential PHI
		calc_PHI(div,x_s,dep,coords,coords_2,phi_all,n,PHI);
		
		// Now implement periodic BCs on PHI
		per_BCs_hor_db(div,x_s,dep,PHI);
		
		// Now we have to exchange the end rows of PHI between the processes
		MPI_Sendrecv(&PHI[div][0][0],d_len,MPI_DOUBLE,below,5,&PHI[0][0][0],d_len,MPI_DOUBLE,above,5,MPI_COMM_WORLD,&status);
		MPI_Sendrecv(&PHI[1][0][0],d_len,MPI_DOUBLE,above,6,&PHI[div+1][0][0],d_len,MPI_DOUBLE,below,6,MPI_COMM_WORLD,&status);
		
		// Copy n to its secondary copy n_2, while setting boundary rows and columns of n_2 to 0
		copy_n(div,x_s,dep,z_s,d_len,d_len_2,n,n_2);
		
		// Calculate what all particles will do and update grid. First for even rows.
		update_all(0,div,x_s,num_As,num_Bs,dep,rot,mu,mu_B,k_m,k_a,k_y,k_b,s_a,s_m,s_b,C_a,C_m,C_b,C_y,del_H,coords,k_dif,PHI,n,n_2);
		
		// Now for odd rows
		update_all(1,div,x_s,num_As,num_Bs,dep,rot,mu,mu_B,k_m,k_a,k_y,k_b,s_a,s_m,s_b,C_a,C_m,C_b,C_y,del_H,coords_2,k_dif,PHI,n,n_2);
		
		// Now n_2 contains the updated particle number array with numbers on the
		// boundaries corresponding to those particles which intend to move there,
		// periodic boundary conditions have NOT yet been applied. Particles on 
		// either horizontal boundary of n_2 must be added to the opposite side.
		// But first we have to communicate the numbers of the new vertical boundary
		// particles.
		MPI_Sendrecv(&n_2[div+1][0][0],d_len,MPI_INT,below,7,&n_strip[0][0],d_len,MPI_INT,above,7,MPI_COMM_WORLD,&status);
		
		// We now have to add the additional particles received from above to those
		// that have migrated from within our own block
		add_strip(x_s,1,dep,n_strip,n_2);
		
		// Now send and receive in the opposite direction
		MPI_Sendrecv(&n_2[0][0][0],d_len,MPI_INT,above,8,&n_strip[0][0],d_len,MPI_INT,below,8,MPI_COMM_WORLD,&status);
		add_strip(x_s,div,dep,n_strip,n_2);
		
		// Add particles that have moved beyond the horizontal edges (periodic BCs)
		fill_Bs(div,x_s,dep,n_2);
		
		// We now have a new array of particle counts, n_2. All we need to do now is copy
		// this new array over the old version, n.
		memcpy(&n[0][0][0],&n_2[0][0][0],d_len_2*sizeof(int));
		
		// Periodically print out the state of the system
		if(t%t_int==0){
			MPI_Gather(&n[1][0][0],d_len_3,MPI_INT,&n_final[0][0][0],div*d_len,MPI_INT,0,MPI_COMM_WORLD);
			if(rank==0){
				add_pr(y_s,x_s,n_final,0,5);							// Membrane particles
				//add_pr(y_s,x_s,n_final,6,8);							// Resource particles
				//add_pr(y_s,x_s,n_final,9,11);							// Waste particles
				add_pr(y_s,x_s,n_final,13,12+num_As);					// A-catalysts
				add_pr(y_s,x_s,n_final,13+num_As,12+num_As+num_Bs);		// B-catalysts
				add_pr(y_s,x_s,n_final,12,12);							// Water
			}
		}
	}
	
	// Calculate final total particle count
	tot=sum_all_procs(div,x_s,dep,n);	
	
	if(rank==0){
		printf("%d\n\n",tot);
	}
	
	MPI_Finalize();
}

