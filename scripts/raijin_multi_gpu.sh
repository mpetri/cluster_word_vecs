[mp9691@raijin1 build]$ cat cluster_w2v_gnews_2gpu.sh
#!/bin/bash

#PBS -P cp1
#PBS -q gpupascal
#PBS -l ncpus=12
#PBS -l ngpus=2
#PBS -l wd
#PBS -l mem=64GB
#PBS -l walltime=40:00:00
#PBS -N C-W2V-GPU2
#PBS -M Matthias.Petri@unimelb.edu.au
#PBS -m abe

BIN_PATH=/home/563/mp9691/cluster_word_vecs/build/
W2V_FILE=/home/563/mp9691/storage/GoogleNews-vectors-negative300.txt

LOG_FILE=/home/563/mp9691/storage/results_cluster/cluster_w32_32k.log

$BIN_PATH/cluster-word-vecs.x \
        -v $W2V_FILE \
        -c 32000 \
        -w 32 \
        -d 0,1 &>> $LOG_FILE
