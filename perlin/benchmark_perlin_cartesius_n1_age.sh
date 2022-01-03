#!/bin/bash
#SBATCH --job-name=benchmark_perlin_n1_age
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p normal 
#SBATCH --constraint=haswell
#SBATCH -t 4-11:00:00 
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
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -a -G -t 365 -N 2**10 -i perlin_CARTESIUS_noMPI_ageP-2pow10_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -a -G -t 365 -N 2**12 -i perlin_CARTESIUS_noMPI_ageP-2pow12_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -a -t 365 -N 2**12 -i perlin_CARTESIUS_noMPI_ageP-2pow12_woGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -a -G -t 365 -N 2**17 -i perlin_CARTESIUS_noMPI_ageP-2pow17_wGC_n1_jit.png
sleep 2
echo '======== SciPy experiments (ONLY ON GEMINI) ========'
echo ' ---- dynamically removing particles (aging with t_max=14 days) ---- '
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -a -G -t 365 -N 2**10 -i perlin_CARTESIUS_noMPI_ageP-2pow10_wGC_n1_scipy.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -a -t 365 -N 2**10 -i perlin_CARTESIUS_noMPI_ageP-2pow10_woGC_n1_scipy.png
sleep 2
