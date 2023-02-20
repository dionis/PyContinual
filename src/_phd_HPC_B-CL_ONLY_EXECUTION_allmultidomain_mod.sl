#!/bin/bash 
#SBATCH --job-name=MODULE_b-cl
##SBATCH --exclusive=user
##SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
##SBATCH --mem=30G
#SBATCH --time=unlimited
#SBATCH --mail-user=dionis@uo.edu.cu
#SBATCH --mail-type=END
#SBATCH -o module_b-cl.%N.%j.out # STDOUT
#SBATCH -e module_b-cl.%N.%j.err # STDERR

module load Python/3.7.0-foss-2018b

cd  $SLURM_SUBMIT_DIR

conda activate phdContinualLearning

 python3 -u run.py --ntasks 8 --bert_model bert-base-uncased --backbone bert_adapter --baseline b-cl --task asc --eval_batch_size 128 --train_batch_size 64 --num_train_epochs 10 --scenario dil_classification --idrandom 0 --dataloaders asc_experiment_phd --use_predefine_args