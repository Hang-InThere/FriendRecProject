import time
from os.path import join

import torch

import Procedure
import register
import utils
import world
from register import dataset

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
torch.autograd.set_detect_anomaly(True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

best_ndcg_1, best_recall_1, best_pre_1 = 0, 0, 0
best_ndcg_2, best_recall_2, best_pre_2 = 0, 0, 0
low_count, low_count_cold = 0, 0
try:
    for epoch in range(world.TRAIN_epochs + 1):
        print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
        start = time.time()
        if epoch % 10 == 1 or epoch == world.TRAIN_epochs:
            print("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, False)
            if results['ndcg'][0] < best_ndcg_1:
                low_count += 1
                if low_count == 30:
                    if epoch > 1000:
                        break
                    else:
                        low_count = 0
            else:
                best_recall_1,best_recall_2= results['recall'][0],results['recall'][1]
                best_ndcg_1,best_ndcg_2 = results['ndcg'][0],results['ndcg'][1]
                best_pre_1,best_pre_2 = results['precision'][0],results['precision'][1]
                low_count = 0


        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
        print(f'[saved][BPR aver loss{loss:.3e}]')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    print(f"\nbest recall at 200:{best_recall_1}")
    print(f"best ndcg at 200:{best_ndcg_1}")
    print(f"best precision at 200:{best_pre_1}")
    print(f"\nbest recall at 300:{best_recall_2}")
    print(f"best ndcg at 300:{best_ndcg_2}")
    print(f"best precision at 300:{best_pre_2}")
