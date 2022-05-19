from xc.libs.data_img import IMGBINDataset, read_raw_img_bin
from xc.libs.custom_dtypes import FeaturesAccumulator
from xc.libs.dataparallel import DataParallel
from xc.models.models_img import Model
import argparse
import torch
import tqdm
import os
# NOTE; Memmap is 2x faster than h5py


def arguments():
    parser = argparse.ArgumentParser(description='Pretrained models')
    parser.add_argument('--data_dir', default=None,
                        help='data directory')
    parser.add_argument('--output_dir', default=None,
                        help='output directory')
    parser.add_argument('--output_file', default=None,
                        help='output directory')
    parser.add_argument('--model', default=None,
                        help='pre trained model directory')
    parser.add_argument('--mode', default=None,
                        help='sub set of data to use')
    parser.add_argument('--b_size', default=100, type=int,
                        help='batch_size')
    parser.add_argument('--resize', default=256, type=int,
                        help='resize shape of the images [always square]')
    args = parser.parse_args()
    if args.model in ["resnet50FPN", "resnet101FPN"]:
        args.b_size = 2
    print(args)
    return args


def collate_fn(batch):
    imgs = torch.cat(list(map(lambda x: x.get_raw_vect(), batch)), dim=0)
    return imgs


if __name__ == '__main__':
    params = arguments()
    batch_size = params.b_size
    resize = params.resize
    params.project_dim = -1
    pre_trained = Model(params.model, params)
    print(pre_trained)
    pre_trained = DataParallel(pre_trained)
    i_file = f"images/{params.mode}.img.bin"
    dataset = IMGBINDataset(params.data_dir, i_file, random_k=-1)
    # dataset.read_func = read_raw_img_bin
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=False, num_workers=6, prefetch_factor=2)
    features = FeaturesAccumulator("Image features", "memmap", ".img.vect")
    mask = None
    with torch.no_grad():
        pre_trained = pre_trained.cuda()
        pre_trained = pre_trained.eval()
        with torch.cuda.amp.autocast():
            for idx, data in enumerate(tqdm.tqdm(dl)):
                embs, mask = pre_trained(data)
                features.transform(embs, mask)
    features.compile()
    print(f"total number of images in datasets are {dataset.data.nnz}")
    """
    Writing the data
    """
    features.remap(dataset.data)
    features.save(os.path.join(params.output_dir, params.mode))
