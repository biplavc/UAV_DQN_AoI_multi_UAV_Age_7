#!/bin/bash

# Example Slurm submission script for Cascades

# NOTE: You will need to edit the Walltime, Resource Request, Queue, and Module lines
# to suit the requirements of your job. You will also, of course have to replace the example job
# commands below with those that run your job.
 
#### Resource Request: ####
# Cascades has the following hardware:
#   a. 190 32-core, 128 GB Intel Broadwell nodes
#   b.   4 32-core, 512 GB Intel Broadwell nodes with 2 Nvidia K80 GPU
#   c.   2 72-core,   3 TB Intel Broadwell nodes
#   d.  39 24-core, 376 GB Intel Skylake nodes with 2 Nvidia V100 GPU
#
# Resources can be requested by specifying the number of nodes, cores, memory, GPUs, etc
# Examples:
#   Request 4 cores (on any number of nodes)
#   #SBATCH --ntasks=4
#   Request exclusive access to all resources on 2 nodes 
#   #SBATCH --nodes=2 
#   #SBATCH --exclusive
#   Request 4 cores (on any number of nodes)
#   #SBATCH --ntasks=4
#   Request 2 nodes with 12 tasks running on each
#   #SBATCH --nodes=2
#   #SBATCH --ntasks-per-node=12
#   Request 12 tasks with 20GB memory per core
#   #SBATCH --ntasks=12 
#   #SBATCH --mem-per-cpu=20G
#   Request 5 nodes and spread 50 tasks evenly across them
#   #SBATCH --nodes=5
#   #SBATCH --ntasks=50
#   #SBATCH --spread-job


###SBATCH --ntasks=1

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G


#### Walltime ####
# Set the walltime, which is the maximum time your job can run in HH:MM:SS
# Note that if your job exceeds the walltime estimated during submission, the scheduler
# will kill it. So it is important to be conservative (i.e., to err on the high side)
# with the walltime that you include in your submission script. 

#SBATCH -t 15:00:00

#### Queue ####
# Queue name. Cascades has five queues:
#   normal_q        for production jobs on all Broadwell nodes
#   largemem_q      for jobs on the two 3TB, 60-core Ivy Bridge servers
#   dev_q           for development/debugging jobs. These jobs must be short but can be large.
#   v100_normal_q   for production jobs on Skylake/V100 nodes
#   v100_dev_q      for development/debugging jobs on Skylake/V100 nodes
#SBATCH -p v100_normal_q

#### Account ####
# This determines which allocation this job's CPU hours are billed to.
# Replace "youraccount" below with the name of your allocation account.
# If you are a student, you will need to get this from your advisor.
# For more on allocations, go here: http://www.arc.vt.edu/allocations
#SBATCH -A vijays

# Add any modules you might require. This example removes all modules and then adds
# the Intel compiler and mvapich2 MPI modules module. Use the module avail command
# to see a list of available modules.
module purge


module load cuda/10.1.168
module load cudnn/7.6.5

echo "SLURM_SUBMIT_DIR is :"
echo $SLURM_SUBMIT_DIR

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Below here enter the commands to start your job. A few examples are provided below.
# Some useful variables set by the job:
#  $SLURM_SUBMIT_DIR   Directory from which the job was submitted
#  $PBS_NODEFILE       File containing list of cores available to the job
#  $PBS_GPUFILE        File containing list of GPUs available to the job
#  $SLURM_JOBID        Job ID (e.g., 107619.master.cluster)
#  $SLURM_NTASKS       Number of cores allocated to the job
# You can run the following (inside a job) to see what environment variables are available:
#  env | grep SLURM
#
# Some useful storage locations (see ARC's Storage documentation for details): 
#  $HOME     Home directory. Use for permanent files.
#  $WORK     Work directory. Use for fast I/O.
#  $TMPFS    File system set up in memory for this job. Use for very fast, small I/O
#  $TMPDIR   Local disk (hard drive) space set up for this job

echo "Starting the execution.."

time /home/biplavc/anaconda3/envs/LabPC_tf/bin/python main_tf.py

# Run the MPI program mpiProg. srun is Slurm's program for running on assigned resources.
#srun ./mpiProg
# The -n flag tells srun how many processes to use. $SLURM_NTASKS is an environment variable
# that holds the number of tasks you requested. So if you selected --nodes=2 and --ntasks-per-node=12
# above, $SLURM_NTASKS will hold 24.
# srun -n $SLURM_NTASKS ./mpiProg

# Run the OpenMP program ompProg with 32 threads (there are 32 cores per Cascades Broadwell node)
#export OMP_NUM_THREADS=32
#./ompProg

exit;

