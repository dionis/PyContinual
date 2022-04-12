#!/bin/bash 

python3 -u run.py --ntasks 8 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_experiment_phd --use_predefine_args --save_model --save_each_step -->> outuput_asc_experiment_phd.txt &
python3 -u run.py --ntasks 8 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_wexperiment_phd --use_predefine_args >> output_asc_wexperiment_phd.txt &
