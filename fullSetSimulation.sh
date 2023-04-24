for i in 32 64 128 256 512
do
	mkdir "N$i"
	cd "N$i"
	../a.out $i straight.config
	mpirun -np 4 python3 simple.py ../in.setup
	cd ..
done
