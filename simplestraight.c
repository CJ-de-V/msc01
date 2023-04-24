/*constructs a polymer and creates a LAMMPS datafile for said polymer
 * this polymer is to be let loose in a constrained sphere so making this
 *  starting configuration small is probably a good idea. How small? not quite
 * clear yet, how much
 * may they overlap? Also not quite clear yet. The soft potentials permit these
 * initial configurations to be disgustingly overlapped. We'll probably setup
 * something that passes for a self avoiding walk that doesn't overlap or tie
 * any funny knots.*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define in(X, Y, Z, W) X * W * W + Y * W + Z
#define Pi 3.14159265359

struct monomer {
        int id;
        double x;
        double y;
        double z;
};


int i=0,j=0,k=0, l=0, dj=1,  dk=1;

/*arguments:
 * number of monomers
 * width of box


 */
int main(int argc, char *argv[]) {
        int N = atoi(argv[1]);  //total number of monomers
        double width =1.0*N;
        double ainc=0.95; //increment size
        double leftlim = 0.025*width;
        //spacing so that diagonal of cube fills 90% of sphere's length along that line.
        //printf("%d monomers => cube width %d\n", N, n);

        struct monomer *listomonomers=(struct monomer*)calloc(N,sizeof(struct monomer));

        for(i=0; i<N; i++) {
                listomonomers[i].id = i+1;
                listomonomers[i].x=leftlim+ainc*i-width/2.0;
                listomonomers[i].y=0.0;
                listomonomers[i].z=0.0;
        }
        //write to tha file
        FILE *configuration = fopen(argv[2],"w");

        fprintf(configuration, "#Arrangement of polymers snaking through a cube orientation\n\n");
        fprintf(configuration, "%8d atoms\n",N );
        fprintf(configuration, "%8d bonds\n",N-1 );
        fprintf(configuration, "%8d angles\n\n",N-2 );
        fprintf(configuration, "%8d atom types\n",1 );
        fprintf(configuration, "%8d bond types\n",1 );
        fprintf(configuration, "%8d angle types\n\n",1 );
        fprintf(configuration, "%8lf %8lf xlo xhi\n",-width/2,width/2 );
        fprintf(configuration, "%8lf %8lf ylo yhi\n",-width/2,width/2 );
        fprintf(configuration, "%8lf %8lf zlo zhi\n\n",-width/2,width/2 );
        fprintf(configuration, "Masses\n\n 1 1\n\n");
        fprintf(configuration, "Atoms\n\n");
        for ( i = 0; i < N; i++) {
                fprintf(configuration,"%8d %8d %8d %8lf %8lf %8lf\n",listomonomers[i].id,1,
                        1,listomonomers[i].x,listomonomers[i].y,listomonomers[i].z );
        }
        fprintf(configuration, "\n Bonds\n\n");
        for ( i = 1; i < N; i++) {
                fprintf(configuration,"%8d %8d %8d %8d\n",i, 1,i,i+1);
        }
        fprintf(configuration, "\n Angles\n\n");
        for ( i = 1; i < N-1; i++) {
                fprintf(configuration,"%8d %8d %8d %8d %8d\n",i, 1,i,i+1,i+2);
        }

        free(listomonomers);
        return 0;
}
