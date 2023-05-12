#!/bin/bash 
#SBATCH --job-name=MODULE_RestMext_2023
##SBATCH --exclusive=user
##SBATCH --partition=public
#SBATCH -p P2 -w c01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
##SBATCH --mem=30G
#SBATCH --time=unlimited
#SBATCH --mail-user=dionis@uo.edu.cu
#SBATCH --mail-type=END
#SBATCH -o module_ctr.%N.%j.out # STDOUT
#SBATCH -e module_ctr.%N.%j.err # STDERR

module load Python/3.7.0-foss-2018b

cd  $SLURM_SUBMIT_DIR

conda activate phdContinualLearning

python3 -u run.py --ntasks 1 --bert_model bertin-project/bertin-gpt-j-6B --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders restmex2022 --use_predefine_args --experiment test_exe --common_prmtrs --save_model --aux_net --save_each_step --num_train_epochs 1
