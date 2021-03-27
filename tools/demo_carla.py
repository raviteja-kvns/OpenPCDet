import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import pickle as pkl

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--mode', type=str, default='pred', help='specify the type of evaluatio - pred or viz')
    parser.add_argument('--viz_file', type=str, default='pred', help='{path of file containing viz data / path to save viz data file}')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    if args.mode == 'viz':
    
        import mayavi.mlab as mlab
        from visual_utils import visualize_utils as V

        with open(args.viz_file, 'rb') as handle:
            viz_data = pkl.load(handle)

        V.draw_scenes(
            points=viz_data['necessary_points'], ref_boxes=viz_data['pred_boxes'],
            ref_scores=viz_data['ref_scores'], ref_labels=viz_data['pred_labels']
        )
        mlab.show(stop=True)

        return 

    else: 

        logger = common_utils.create_logger()
        logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), ext=args.ext, logger=logger
        )
        logger.info(f'Total number of samples: \t{len(demo_dataset)}')

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)

                viz_data = {}
                viz_data['necessary_points'] = data_dict['points'][:, 1:].cpu()
                viz_data['pred_boxes'] = pred_dicts[0]['pred_boxes'].cpu()
                viz_data['ref_scores'] = pred_dicts[0]['pred_scores'].cpu()
                viz_data['pred_labels'] = pred_dicts[0]['pred_labels'].cpu()

                with open(args.viz_file, 'wb') as handle:
                    pkl.dump(viz_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
                    print("Successfully saved pickle file")

        logger.info('Demo done.')


if __name__ == '__main__':
    main()
