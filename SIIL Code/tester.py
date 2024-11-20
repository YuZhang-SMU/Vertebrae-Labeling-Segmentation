import torch

import os

import torch.utils.data as Data
import torchio
from tqdm import tqdm

from get_data import  get_data_test
from utilize import print_object

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings("ignore")

from get_config import MyConfig
from models.siil import SIIL

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    myconfig = MyConfig()
    print_object(myconfig)
    print('--------------  Load_Model  --------------')

    # network
    mynetwork = SIIL(myconfig)
    load_model = torch.load(myconfig.latest_model)
    mynetwork.load_state_dict(load_model.state_dict(), strict=True)
    print('Model already exist, load model Successful')
    mynetwork = mynetwork.to(my_device)

    # data
    test_dataset = get_data_test(myconfig)

    # test
    mynetwork.eval()
    for idx, subj in enumerate(test_dataset.subjects):
        name = subj['image']['stem']
        grid_sampler = torchio.inference.GridSampler(
            subject=subj,
            patch_size=(64, 64, 128),
            patch_overlap=(32, 32, 64)
        )
        patch_loader = Data.DataLoader(grid_sampler, batch_size=4)
        aggregator1 = torchio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
        aggregator2 = torchio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                locations = patches_batch[torchio.LOCATION]
                output_dict = mynetwork(data=patches_batch, test=True)
                aggregator1.add_batch(output_dict['seg_one_final'].permute(0, 1, 3, 4, 2), locations)
                aggregator2.add_batch(output_dict['seg_multi_final'].permute(0, 1, 3, 4, 2), locations)

        output1_logit = aggregator1.get_output_tensor()
        output2_logit = aggregator2.get_output_tensor()
        output2_probs = torch.softmax(output2_logit, dim=0)
        output2_probs_np = output2_probs.cpu().numpy()
        output2_region = torch.sigmoid(output1_logit).round() * torch.sigmoid(output2_logit).round()
        print(output2_region.shape)
