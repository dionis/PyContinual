#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ls


# In[ ]:


cd PHD_Experiment/PyContinual/src


# In[ ]:


get_ipython().system('python run.py --ntasks 3 --bert_model bert-base-uncased --backbone bert_adapter --baseline ar1 --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders test_dataset_execution --use_predefine_args --num_train_epochs 1 --experiment test_exe --common_prmtrs --save_model --aux_net --save_each_step --local_execution')


# In[ ]:


get_ipython().system('git commit -am "Update"')


# In[ ]:


get_ipython().system('git pull --force')


# In[ ]:





# In[ ]:




