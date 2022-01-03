#!/bin/bash
#SBATCH --job-name=benchmark_perlin_n1_const
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
echo ' ---- constant particle number ---- '
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -t 365 -G -N 2**10 -i perlin_CARTESIUS_noMPI_constP-2pow10_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -t 365 -G -N 2**12 -i perlin_CARTESIUS_noMPI_constP-2pow12_wGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -t 365 -N 2**12 -i perlin_CARTESIUS_noMPI_constP-2pow12_woGC_n1_jit.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -t 365 -G -N 2**17 -i perlin_CARTESIUS_noMPI_constP-2pow17_wGC_n1_jit.png
sleep 2
echo '======== SciPy experiments (ONLY ON GEMINI) ========'
echo ' ---- constant particle number ---- '
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -t 365 -G -N 2**10 -i perlin_CARTESIUS_noMPI_constP-2pow10_wGC_n1_scipy.png
sleep 2
python ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -t 365 -N 2**10 -i perlin_CARTESIUS_noMPI_constP-2pow10_woGC_n1_scipy.png
sleep 2
