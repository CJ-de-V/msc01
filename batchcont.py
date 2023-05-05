# batchcontrun.py
# Purpose: adapted from batchrun
# Should do the exact same thing as batchrun, except running from restart files is slightly different.

# Serial syntax: batchcont.py ???
#                ??????

# Parallel syntax: mpirun -np 4 python3 batchcont.py ???
#                  in.lammps = LAMMPS input script

from __future__ import print_function
import sys
import pandas as pd
import ctypes
from mpi4py import MPI
from lammps import lammps

# parse command line

argv = sys.argv
if len(argv) != 2:
    print("Syntax: simple.py in.lammps")
    sys.exit()

infile = sys.argv[1]

me = 0
me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

# N = [32, 64, 128, 256, 512]
# K = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]

# List of N values along with which K values need to be run for a longer time.
runmelonger = [[32, [32, 64, 128]],
               [64, [16, 32, 64, 128]],
               [128, [16]],
               [256, [0.25, 0.5, 16]],
               [512, [0.25, 0.5, 1, 16]]]

# run infile one line at a time

#adapt from continuedrestart?

for nray in runmelonger:
    for k in nray[1]:
        n = nray[0]

        lmp=lammps()
        #read in which restart file we're working with
        lmp.command('read_restart restart._N'+n+'_k'+k+'.dat')

        lines = open(infile, 'r').readlines()
        for line in lines: lmp.command(line)
        
        # setup commands that run in in.continuedrestart here, computes and variables mostly
        # dumps are already below since they were named in the python script to begin with

        initialize

        # setup fixes that write to file, Thermo, R_g, and dump file, read in restart file
        fixsetup = ["fix mythermofile all print 10000 \"$t ${mytemp} ${myepair}\" file thermo_output_N" + str(
            n) + "_k" + str(k) + ".dat screen no",
                    # thermodynamic data outputted to the file appropriately named (sort of)
                    "fix myRG2file all print 10000 \"$t ${RG2}\" file radius_of_gyration_squared_N" + str(
                        n) + "_k" + str(k) + ".dat screen no",
                    "dump dum2 all custom 10000 dump_N" + str(n) + "_k" + str(k) + ".dynamics id type x y z",
                    "dump_modify dum2  sort id"
                    ]
        lmp.commands_list(fixsetup)

        # run and collect data under these new fixes
        lmp.command("run 250000")

        # remove fixes
        cleansetup = ["write_restart 	restart._N" + str(monomercount) + "_k" + str(k) + ".dat",
                      "unfix mythermofile",
                      "unfix myRG2file",
                      "undump dum2"]
        lmp.commands_list(cleansetup)

    # write restart file, possibly reset timestep? seems useless to do so ngl

# uncomment if running in parallel via mpi4py
print("Proc %d out of %d procs has" % (me, nprocs), lmp)

MPI.COMM_WORLD.Barrier()
# Definitely worth looking at the for throwing these in the for loops, might be that only one core is ending the dumps before the others get to dumping
MPI.Finalize()
