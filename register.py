from pprint import pprint
import time


import Dataset1
import model
import world


if world.dataset in ['lastfm', 'ciao', 'epinions', 'douban', 'gowalla']:
    if world.model_name in ['FriendRec']:
        dataset = Dataset1.SocialDataset(world.dataset)


print('===========config================')
pprint(world.config)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {

    'LightGCN': model.LightGCN,
    'FriendRec': model.FriendRec,
}
