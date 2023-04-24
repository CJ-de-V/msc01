# simple.py
# Purpose: adapted from /python/examples/simple.py
#should iterate over different k values and write out data to properly labelled datafiles

# Serial syntax: simple.py in.lammps
#                in.lammps = LAMMPS input script

# Parallel syntax: mpirun -np 4 python3 simple.py in.lammps
#                  in.lammps = LAMMPS input script

from __future__ import print_function
import sys
import ctypes

# parse command line

argv = sys.argv
if len(argv) != 2:
  print("Syntax: simple.py in.lammps")
  sys.exit()

infile = sys.argv[1]

me = 0

# uncomment this if running in parallel via mpi4py
from mpi4py import MPI
me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

from lammps import lammps
lmp = lammps()

# run infile one line at a time

lines = open(infile,'r').readlines()
for line in lines: lmp.command(line)

K=[0.25,0.5,1,2,4,8,16,32,64,128]
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>NUMBER OF ATOMS="+str(lmp.get_natoms()))
monomercount=lmp.get_natoms();
for k in K:
    lmp.command("angle_coeff   1  "+str(k))         #increase bendingcost
    lmp.command("print \">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>k is "+str(k)+" \" ")
    lmp.command("run 10000")                        #adapt to new bending cost

    #setup fixes that write to file, Thermo, R_g, and dump file
    fixsetup=["fix mythermofile all print 10000 \"$t ${mytemp} ${myepair}\" file thermo_output_N"+str(monomercount)+"_k"+str(k)+".dat screen no",
              #thermodynamic data outputted to the file appropriately named (sort of)
              "fix myRG2file all print 10000 \"$t ${RG2}\" file radius_of_gyration_squared_N"+str(monomercount)+"_k"+str(k)+".dat screen no",
              "dump dum2 all custom 10000 dump_N"+str(monomercount)+"_k"+str(k)+".dynamics id type x y z",
              "dump_modify dum2  sort id"
              ]
    lmp.commands_list(fixsetup)

    #run and collect data under these new fixes
    lmp.command("run 250000")


    #remove fixes
    cleansetup=["write_restart 	restart._N"+str(monomercount)+"_k"+str(k)+".dat",
    		    "unfix mythermofile",
                "unfix myRG2file",
                "undump dum2"]
    lmp.commands_list(cleansetup)

    # write restart file, possibly reset timestep? seems useless to do so ngl


# uncomment if running in parallel via mpi4py
print("Proc %d out of %d procs has" % (me,nprocs), lmp)

MPI.COMM_WORLD.Barrier()
#Definitely worth looking at the for throwing these in the for loops, might be that only one core is ending the dumps before the others get to dumping
MPI.Finalize()
