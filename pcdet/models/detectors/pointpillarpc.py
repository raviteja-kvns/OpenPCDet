import torch

from .pointpillar import PointPillar

class PointPillarPC(PointPillar):

    def __init__(self, model_cfg, num_class, dataset):

        super().__init__(model_cfg, num_class, dataset)

        # Adding the voxel to point layer
        # self.vox_to_point = torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)

        # Adding MLP to pass these point features


    def forward(self, batch_dict):

        module_count = len(self.module_list)

        # Excluding module count to avoid the last module that is not needed which computes OpenPCDet losses
        cnt = 0
        for cur_module in self.module_list:
            if cnt == module_count - 1:
                break
            batch_dict = cur_module(batch_dict)
            cnt += 1

        # Fwd Pass through new voxel to point layer
        # self.vox_to_point()
        output = torch.nn.functional.grid_sample(input = batch_dict['spatial_features_2d'],
                                                grid = batch_dict['points'][:, 0:2],
                                                mode = 'bilinear', 
                                                padding_mode = 'zeros', 
                                                align_corners = None)

        # Concatenate z co-ordinates

        # Fwd pass through MLP to get features 

        a=1

        # Commenting out this since we want to train this with Point Contrast loss
        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss()

        #     ret_dict = {
        #         'loss': loss
        #     }
        #     return ret_dict, tb_dict, disp_dict
        # else:
        
        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return pred_dicts, recall_dicts