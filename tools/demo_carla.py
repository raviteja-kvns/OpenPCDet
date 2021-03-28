import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import pickle as pkl

def parse_config(parse_cfg=True, cfg=None, cfg_from_yaml_file=None):
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

    if parse_cfg:
        cfg_from_yaml_file(args.cfg_file, cfg)
    else:
        print("Skipping: Parse Config")
        cfg = {}

    return args, cfg


def main():
    args, _ = parse_config(parse_cfg=False)

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

        from pcdet.config import cfg
        from pcdet.config import cfg_from_yaml_file
        from pcdet.datasets import DemoDataset
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils

        args, cfg = parse_config(parse_cfg=True, cfg=cfg, cfg_from_yaml_file=cfg_from_yaml_file)

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
