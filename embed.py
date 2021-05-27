import argparse
import torch
import utils
import os
import pickle


from torch.utils import data
import numpy as np
from tqdm.auto import tqdm

import modules
import utils

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
                    default='checkpoints/shapes',
                    help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=25,
                    help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_eval.h5',
                    help='Dataset string.')
parser.add_argument('--fname', type=str,
                    default='data/embed_eval.h5',
                    help='Embedding location.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')

args_eval = parser.parse_args()


meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')

args = pickle.load(open(meta_file, 'rb'))['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 100
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.PathShapeDataset(hdf5_file=args.dataset, path_length=args_eval.num_steps)
data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Get data sample
obs = data_loader.__iter__().next()[0]
input_shape = obs[0][0].size()

model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    encoder=args.encoder).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

pred_states = []

with torch.no_grad():

    for batch_idx, data_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        data_batch = [[t.to(
            device) for t in tensor] for tensor in data_batch]
        observations, actions, shapes = data_batch

        if observations[0].size(0) != args.batch_size:
            continue

        obs = observations[0]

        state = model.obj_encoder(model.obj_extractor(obs))

        pred_state = state
        for i in range(args_eval.num_steps):
            pred_trans = model.transition_model(pred_state, actions[i])
            pred_state = pred_state + pred_trans
            pred_states.append({
                'embed': pred_state.cpu(),
                'shapes': shapes[i].cpu(),
                })

utils.save_list_dict_h5py(pred_states, args_eval.fname)
