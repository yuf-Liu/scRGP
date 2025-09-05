import sys
from Training import scRGP
from PertDataProcess import PertData

import time

start = time.time()
data_name = sys.argv[1]
seed = sys.argv[2]

# available_splits = ['combo_seen0', 'combo_seen1', 'combo_seen2', 'single', 'no_test', 'only_test', 'only_train', 'load_split']
# task: gene_pert, cross_cells, cell_type_switch 
pert_data = PertData('/data4/yfliu/single_cell/my_work/data')
pert_data.load(data_name=data_name, mode='')  # specific dataset name
pert_data.prepare_split(split='only_train', seed=int(seed))  # get data split with seed
pert_data.get_dataloader(batch_size=32, test_batch_size=32)  # prepare data loader

model = scRGP(pert_data, device='cuda:2')
model.model_initialize(hidden_size=64, no_perturb=False)

model.train(epochs=6, lr=1e-3)
model.eval_model('replogle_k562_only_train/'+data_name+seed+'.csv')

model.save_model('replogle_k562_only_train/model'+seed+'/')

#model.load_pretrained('/data5/yfliu/single_cell/my_work/rewrite/norman_single')
#model.predict('/data5/yfliu/single_cell/my_work/data/norman/perturb_processed.h5ad')
#gears_model.eval_model()

end = time.time()
print(end-start)
