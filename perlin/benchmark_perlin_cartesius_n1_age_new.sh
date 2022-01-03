#!/bin/bash
#SBATCH --job-name=benchmark_perlin_n1_age_new
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p normal 
#SBATCH --constraint=haswell
#SBATCH -t 5-00:00:00 
#SBATCH --mem=48000 
#SBATCH --mail-type=END

#SBATCH --mail-user=<fill-in-mail-address>

#
# Here the shell script starts. 
# Go to the working directory:
#

echo 'Initialize environment'

export SCRIPT_LOCATION = #<location-of-these-runscripts>#
cd ${SCRIPT_LOCATION}
export PARCELS_HEAD= #<location your your parcels checkout from github; main folder>#
export TARGET_HEAD= #<location where your benchmark results shall be stored>#

echo '======== JIT (Just-in-Time) experiments - single-core ========'
echo ' ---- dynamically removing particles (aging with t_max=14 days) ---- '
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -G -a -t 365 -N 32*1024 -i perlin_CARTESIUS_noMPI_ageP-32k_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -G -a -t 365 -N 64*1024 -i perlin_CARTESIUS_noMPI_ageP-64k_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -G -a -t 365 -N 96*1024 -i perlin_CARTESIUS_noMPI_ageP-96k_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -G -a -t 365 -N 128*1024 -i perlin_CARTESIUS_noMPI_ageP-128k_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -G -a -t 365 -N 192*1024 -i perlin_CARTESIUS_noMPI_ageP-192k_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -G -a -t 365 -N 256*1024 -i perlin_CARTESIUS_noMPI_ageP-256k_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -G -a -t 365 -N 320*1024 -i perlin_CARTESIUS_noMPI_ageP-320k_wGC_n1_jit.png
sleep 2
# python ${PARCELS_HEAD}/performance/benchmark_perlin.py -G -a -t 365 -N 384*1024 -i perlin_CARTESIUS_noMPI_ageP-384k_wGC_n1_jit.png
# sleep 2
