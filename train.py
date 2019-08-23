import argparse
import datetime
import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.distance import PairwiseDistance
from torch.optim import lr_scheduler

from data_loader import get_dataloader
from datasets.write_csv_for_making_dataset import write_csv
from eval_metrics import evaluate, plot_roc
from loss import TripletLoss
from models import FaceNetModel
from utils import ModelSaver, init_log_just_created

parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument('--num-epochs', default=200, type=int, metavar='NE',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--num-train-triplets', default=10000, type=int, metavar='NTT',
                    help='number of triplets for training (default: 10000)')
parser.add_argument('--num-valid-triplets', default=10000, type=int, metavar='NVT',
                    help='number of triplets for vaidation (default: 10000)')
parser.add_argument('--batch-size', default=64, type=int, metavar='BS',
                    help='batch size (default: 128)')
parser.add_argument('--num-workers', default=8, type=int, metavar='NW',
                    help='number of workers (default: 8)')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--margin', default=0.5, type=float, metavar='MG',
                    help='margin (default: 0.5)')
parser.add_argument('--train-root-dir', default='/home/khairulimam/datasets/vggv2/test-mtcnn-182/', type=str,
                    help='path to train root dir')
parser.add_argument('--valid-root-dir', default='/home/khairulimam/datasets/lfw-mtcnn-182/', type=str,
                    help='path to valid root dir')
parser.add_argument('--train-csv-name', default='datasets/test-vggv2-mtcnn-182.csv', type=str,
                    help='list of training images')
parser.add_argument('--valid-csv-name', default='datasets/lfw-mtcnn-182.csv', type=str, help='list of validtion images')
parser.add_argument('--step-size', default=50, type=int, metavar='SZ',
                    help='Decay learning rate schedules every --step-size (default: 50)')
parser.add_argument('--unfreeze', type=str, metavar='UF', default='',
                    help='Provide an option for unfreezeing given layers')
parser.add_argument('--freeze', type=str, metavar='F', default='',
                    help='Provide an option for freezeing given layers')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--fc-only', action='store_true')
parser.add_argument('--except-fc', action='store_true')
parser.add_argument('--load-best', action='store_true')
parser.add_argument('--load-last', action='store_true')
parser.add_argument('--continue-step', action='store_true')
parser.add_argument('--train-all', action='store_true', help='Train all layers')

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)
modelsaver = ModelSaver()


def save_if_best(state, acc):
    modelsaver.save_if_best(acc, state)


def main():
    init_log_just_created("log/valid.csv")
    init_log_just_created("log/train.csv")
    import pandas as pd
    valid = pd.read_csv('log/valid.csv')
    max_acc = valid['acc'].max()

    pretrain = args.pretrain
    fc_only = args.fc_only
    except_fc = args.except_fc
    train_all = args.train_all
    unfreeze = args.unfreeze.split(',')
    freeze = args.freeze.split(',')
    start_epoch = 0
    print(f"Transfer learning: {pretrain}")
    print("Train fc only:", fc_only)
    print("Train except fc:", except_fc)
    print("Train all layers:", train_all)
    print("Unfreeze only:", ', '.join(unfreeze))
    print("Freeze only:", ', '.join(freeze))
    print(f"Max acc: {max_acc:.4f}")
    print(f"Learning rate will decayed every {args.step_size}th epoch")
    model = FaceNetModel(pretrained=pretrain)
    model.to(device)
    triplet_loss = TripletLoss(args.margin).to(device)

    if fc_only:
        model.unfreeze_only(['fc', 'classifier'])
    if except_fc:
        model.freeze_only(['fc', 'classifier'])
    if train_all:
        model.unfreeze_all()
    if len(unfreeze) > 0:
        model.unfreeze_only(unfreeze)
    if len(freeze) > 0:
        model.freeze_only(freeze)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    if args.load_best or args.load_last:
        checkpoint = './log/best_state.pth' if args.load_best else './log/last_checkpoint.pth'
        print('loading', checkpoint)
        checkpoint = torch.load(checkpoint)
        modelsaver.current_acc = max_acc
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        print("Stepping scheduler")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        except ValueError as e:
            print("Can't load last optimizer")
            print(e)
        if args.continue_step:
            scheduler.step(checkpoint['epoch'])
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}\n"
              f"Loaded checkpoint accuracy: {checkpoint['accuracy']}\n"
              f"Loaded checkpoint loss: {checkpoint['loss']}")

    model = torch.nn.DataParallel(model)

    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, args.num_epochs + start_epoch - 1))

        time0 = time.time()
        data_loaders, data_size = get_dataloader(args.train_root_dir, args.valid_root_dir,
                                                 args.train_csv_name, args.valid_csv_name,
                                                 args.num_train_triplets, args.num_valid_triplets,
                                                 args.batch_size, args.num_workers)

        train_valid(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size)
        print(f'  Execution time                 = {time.time() - time0}')
    print(80 * '=')


def save_last_checkpoint(state):
    torch.save(state, 'log/last_checkpoint.pth')


def train_valid(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size):
    for phase in ['train', 'valid']:

        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            scheduler.step()
            if scheduler.last_epoch % scheduler.step_size == 0:
                print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)

            # pos_cls = batch_sample['pos_class'].to(device)
            # neg_cls = batch_sample['neg_class'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                anc_hard_img = anc_img[hard_triplets]
                pos_hard_img = pos_img[hard_triplets]
                neg_hard_img = neg_img[hard_triplets]

                # pos_hard_cls = pos_cls[hard_triplets]
                # neg_hard_cls = neg_cls[hard_triplets]

                model.module.forward_classifier(anc_hard_img)
                model.module.forward_classifier(pos_hard_img)
                model.module.forward_classifier(neg_hard_img)

                triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()

                distances.append(pos_dist.data.cpu().numpy())
                labels.append(np.ones(pos_dist.size(0)))

                distances.append(neg_dist.data.cpu().numpy())
                labels.append(np.zeros(neg_dist.size(0)))

                triplet_loss_sum += triplet_loss.item()

        avg_triplet_loss = triplet_loss_sum / data_size[phase]
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = '_'.join(map(str, scheduler.get_lr()))
        layers = '+'.join(args.unfreeze.split(','))
        write_csv(f'log/{phase}.csv', [time, epoch, np.mean(accuracy), avg_triplet_loss, layers, args.batch_size, lr])

        if phase == 'valid':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.module.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'accuracy': np.mean(accuracy),
                                  'loss': avg_triplet_loss
                                  })
            save_if_best({'epoch': epoch,
                          'state_dict': model.module.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'accuracy': np.mean(accuracy),
                          'loss': avg_triplet_loss
                          }, np.mean(accuracy))
        else:
            plot_roc(fpr, tpr, figure_name='./log/roc_valid_epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    main()
