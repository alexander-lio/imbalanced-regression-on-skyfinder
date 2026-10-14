import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from resnet import resnet50
from loss import *
from datasets import AgeDB
from utils import *

import os
os.environ["KMP_WARNINGS"] = "FALSE"

# CHANGED: label offset constant — predictions are in label-space; subtract to get C
LABEL_OFFSET = 30


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# imbalanced related
# LDS
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
# FDS
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=9, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=81, help='maximum bucket considered for FDS')  # CHANGED: 100->81
parser.add_argument('--bucket_start', type=int, default=0, choices=[0, 3],
                    help='minimum(starting) bucket for FDS, 0 for SkyFinder/IMDBWIKI, 3 for AgeDB')  # CHANGED: default 3->0
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'], help='cost-sensitive reweighting scheme')
# two-stage training: RRT
parser.add_argument('--retrain_fc', action='store_true', default=False, help='whether to retrain last regression layer (regressor)')

# training/optimization related
parser.add_argument('--dataset', type=str, default='skyfinder',
                    choices=['imdb_wiki', 'agedb', 'skyfinder'], help='dataset name')  # CHANGED: added skyfinder, new default
parser.add_argument('--data_dir', type=str, default='../data/images', help='directory containing <CamId>/<Filename> images')  # CHANGED: points to images dir
parser.add_argument('--csv', type=str, default='../data/skyfinder.csv', help='path to the split CSV')  # CHANGED: new arg
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=90, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')
# checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')
# CHANGED: new args
parser.add_argument('--pretrained_imagenet', action='store_true', default=True,
                    help='init backbone with torchvision ImageNet weights (default: on)')
parser.add_argument('--no_pretrained_imagenet', dest='pretrained_imagenet', action='store_false',
                    help='disable ImageNet pretraining')
parser.add_argument('--subset_frac', type=float, default=1.0,
                    help='fraction of each split to use (for quick smoke tests)')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

args.start_epoch, args.best_loss = 0, 1e5

# CHANGED: device handling — support cuda, mps, cpu (original hard-coded .cuda())
if args.gpu is not None and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
elif torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_fc:
    args.store_name += f'_retrain_fc'
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"

prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")
print(f"Device: {device}")  # CHANGED

# CHANGED: tensorboard_logger is optional (not always installed)
try:
    from tensorboard_logger import Logger
    tb_logger = Logger(logdir=os.path.join(args.store_root, args.store_name), flush_secs=2)
except ImportError:
    tb_logger = None
    print("tensorboard_logger not installed, skipping TB logging")


def main():
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # Data
    print('=====> Preparing data...')
    print(f"CSV: {args.csv}")  # CHANGED: use --csv instead of data_dir/{dataset}.csv
    df = pd.read_csv(args.csv)
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']

    # CHANGED: subset for quick smoke tests
    if args.subset_frac < 1.0:
        df_train = df_train.sample(frac=args.subset_frac, random_state=42).reset_index(drop=True)
        df_val = df_val.sample(frac=args.subset_frac, random_state=42).reset_index(drop=True)
        df_test = df_test.sample(frac=args.subset_frac, random_state=42).reset_index(drop=True)
        print(f"Using subset_frac={args.subset_frac}: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    train_labels = df_train['label']  # CHANGED: 'age' -> 'label'

    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train',
                          reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val')
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == 'cuda'), drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == 'cuda'), drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=(device.type == 'cuda'), drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('=====> Building model...')
    model = resnet50(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                     start_update=args.start_update, start_smooth=args.start_smooth,
                     kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)

    # CHANGED: load torchvision ImageNet weights into the repo's resnet (before wrapping)
    if args.pretrained_imagenet and not args.pretrained and not args.resume:
        import torchvision.models as tv_models
        tv_state = tv_models.resnet50(weights='IMAGENET1K_V1').state_dict()
        # Map torchvision keys to repo keys; skip fc (different output size) and linear
        mapped, skipped = [], []
        model_keys = set(model.state_dict().keys())
        repo_state = {}
        for k, v in tv_state.items():
            # torchvision uses 'fc', repo uses 'linear' — skip both since sizes differ
            if k.startswith('fc.'):
                skipped.append(k)
                continue
            if k in model_keys and v.shape == model.state_dict()[k].shape:
                repo_state[k] = v
                mapped.append(k)
            else:
                skipped.append(k)
        missing = model.load_state_dict(repo_state, strict=False)
        print(f"===> ImageNet pretrained: loaded {len(mapped)} keys, "
              f"skipped {len(skipped)} ({skipped}), "
              f"missing in checkpoint: {missing.missing_keys}")

    # CHANGED: use device instead of DataParallel+cuda; wrap in DataParallel only for multi-GPU
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)
        use_data_parallel = True
    else:
        model = model.to(device)
        use_data_parallel = False

    # evaluate only
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        validate(test_loader, model, train_labels=train_labels, prefix='Test',
                 df_eval=df_test)  # CHANGED: pass df for pred saving
        return

    if args.retrain_fc:
        assert args.reweight != 'none' and args.pretrained
        print('===> Retrain last regression layer only!')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'linear' not in name:
                param.requires_grad = False

    # Loss and optimizer
    if not args.retrain_fc:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # optimize only the last linear layer
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        mod = model.module if use_data_parallel else model  # CHANGED: handle non-DataParallel
        names = list(filter(lambda k: k is not None, [k if v.requires_grad else None for k, v in mod.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)  # CHANGED: use device
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    if device.type == 'cuda':
        cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(train_loader, model, optimizer, epoch)
        val_loss_mse, val_loss_l1, val_loss_gmean = validate(val_loader, model, train_labels=train_labels)

        loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best {'L1' if 'l1' in args.loss else 'MSE'} Loss: {args.best_loss:.3f}")
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

        if tb_logger:  # CHANGED: guard for missing tensorboard_logger
            tb_logger.log_value('train_loss', train_loss, epoch)
            tb_logger.log_value('val_loss_mse', val_loss_mse, epoch)
            tb_logger.log_value('val_loss_l1', val_loss_l1, epoch)
            tb_logger.log_value('val_loss_gmean', val_loss_gmean, epoch)

    # test with best checkpoint
    print("=" * 120)
    print("Test best model on testset...")
    best_path = f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar"
    checkpoint = torch.load(best_path, map_location=device)  # CHANGED
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")
    test_loss_mse, test_loss_l1, test_loss_gmean = validate(
        test_loader, model, train_labels=train_labels, prefix='Test',
        df_eval=df_test)  # CHANGED: pass df for pred saving
    print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter(f'Loss ({args.loss.upper()})', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs = inputs.to(device, non_blocking=True)      # CHANGED: .cuda() -> .to(device)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        if args.fds:
            outputs, _ = model(inputs, targets, epoch)
        else:
            outputs = model(inputs, targets, epoch)

        loss = globals()[f"weighted_{args.loss}_loss"](outputs, targets, weights)
        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for (inputs, targets, _) in tqdm(train_loader):
                inputs = inputs.to(device, non_blocking=True)  # CHANGED
                outputs, feature = model(inputs, targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings = torch.from_numpy(np.vstack(encodings)).to(device)  # CHANGED
        labels = torch.from_numpy(np.hstack(labels)).to(device)
        mod = model.module if hasattr(model, 'module') else model  # CHANGED
        mod.FDS.update_last_epoch_stats(epoch)
        mod.FDS.update_running_stats(encodings, labels, epoch)

    return losses.avg


def validate(val_loader, model, train_labels=None, prefix='Val', df_eval=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels = [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs = inputs.to(device, non_blocking=True)   # CHANGED
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)

            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)

        preds_flat = np.hstack(preds)
        labels_flat = np.hstack(labels)

        shot_dict = shot_metrics(preds_flat, labels_flat, train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        # CHANGED: all metrics are already in label-space; MAE is shift-invariant so it equals C
        print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 (MAE C) {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
              f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")

        # CHANGED: temperature-range group metrics (<-10 C, -10..35 C, >35 C)
        temp_range_metrics(preds_flat, labels_flat, prefix)

        # CHANGED: save test predictions to CSV
        if prefix == 'Test' and df_eval is not None:
            save_test_predictions(preds_flat, labels_flat, df_eval)

    return losses_mse.avg, losses_l1.avg, loss_gmean


# CHANGED: new function — MAE by temperature range
def temp_range_metrics(preds, labels, prefix='Val'):
    """Report MAE for three temperature ranges: <-10 C, -10..35 C, >35 C."""
    # Convert from label-space to C
    preds_c = preds - LABEL_OFFSET
    labels_c = labels - LABEL_OFFSET

    ranges = [
        ('< -10 C',  labels_c < -10),
        ('-10..35 C', (labels_c >= -10) & (labels_c <= 35)),
        ('> 35 C',    labels_c > 35),
    ]
    parts = []
    for name, mask in ranges:
        if mask.sum() > 0:
            mae = np.mean(np.abs(preds_c[mask] - labels_c[mask]))
            parts.append(f"{name}: MAE {mae:.3f} (n={mask.sum()})")
        else:
            parts.append(f"{name}: n/a (n=0)")
    print(f" * {prefix} temp-range: " + " | ".join(parts))


# CHANGED: new function — save per-image test predictions
def save_test_predictions(preds, labels, df_eval):
    """Save test predictions to results/<store_name>/test_preds.csv."""
    results_dir = os.path.join('results', args.store_name)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'test_preds.csv')

    df_out = df_eval[['path', 'CamId', 'is_night']].copy().reset_index(drop=True)
    df_out['true_temp_c'] = labels.flatten() - LABEL_OFFSET
    df_out['pred_temp_c'] = preds.flatten() - LABEL_OFFSET
    df_out.to_csv(out_path, index=False)
    print(f"===> Test predictions saved to {out_path} ({len(df_out)} rows)")


def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    # CHANGED: guard against empty groups (possible with small subsets)
    for name, mse_list, l1_list, gmean_list, cnt_list in [
        ('many', many_shot_mse, many_shot_l1, many_shot_gmean, many_shot_cnt),
        ('median', median_shot_mse, median_shot_l1, median_shot_gmean, median_shot_cnt),
        ('low', low_shot_mse, low_shot_l1, low_shot_gmean, low_shot_cnt),
    ]:
        if cnt_list and np.sum(cnt_list) > 0:
            shot_dict[name]['mse'] = np.sum(mse_list) / np.sum(cnt_list)
            shot_dict[name]['l1'] = np.sum(l1_list) / np.sum(cnt_list)
            shot_dict[name]['gmean'] = gmean(np.hstack(gmean_list), axis=None).astype(float) if gmean_list else 0.0
        else:
            shot_dict[name]['mse'] = 0.0
            shot_dict[name]['l1'] = 0.0
            shot_dict[name]['gmean'] = 0.0

    return shot_dict


if __name__ == '__main__':
    main()
