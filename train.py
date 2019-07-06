import argparse
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
from utils import ModelSaver, create_if_not_exist, init_log_just_created

parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument('--num-epochs', default=200, type=int, metavar='NE',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--num-classes', default=10000, type=int, metavar='NC',
                    help='number of clases (default: 10000)')
parser.add_argument('--num-train-triplets', default=10000, type=int, metavar='NTT',
                    help='number of triplets for training (default: 10000)')
parser.add_argument('--num-valid-triplets', default=10000, type=int, metavar='NVT',
                    help='number of triplets for vaidation (default: 10000)')
parser.add_argument('--embedding-size', default=128, type=int, metavar='ES',
                    help='embedding size (default: 128)')
parser.add_argument('--batch-size', default=64, type=int, metavar='BS',
                    help='batch size (default: 128)')
parser.add_argument('--num-workers', default=8, type=int, metavar='NW',
                    help='number of workers (default: 8)')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--margin', default=0.5, type=float, metavar='MG',
                    help='margin (default: 0.5)')
parser.add_argument('--train-root-dir',
                    default='/run/media/hoosiki/WareHouse2/home/mtb/datasets/vggface2/test_mtcnnpy_182', type=str,
                    help='path to train root dir')
parser.add_argument('--valid-root-dir', default='/run/media/hoosiki/WareHouse2/home/mtb/datasets/lfw/lfw_mtcnnpy_182',
                    type=str,
                    help='path to valid root dir')
parser.add_argument('--train-csv-name', default='./datasets/test_vggface2.csv', type=str,
                    help='list of training images')
parser.add_argument('--valid-csv-name', default='./datasets/lfw.csv', type=str,
                    help='list of validtion images')
parser.add_argument('--step-size', default=50, type=int, metavar='SZ',
                    help='Decay learning rate schedules every --step-size (default: 50)')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--fc-only', action='store_true')
parser.add_argument('--except-fc', action='store_true')
parser.add_argument('--load-best', action='store_true')
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

    pretrain = args.pretrain
    fc_only = args.fc_only
    except_fc = args.except_fc
    train_all = args.train_all
    start_epoch = 0
    print(f"Transfer learning: {pretrain}")
    print("Train fc only:", fc_only)
    print("Train except fc:", except_fc)
    print("Train all layers:", train_all)
    print(f"Learning rate will decayed every {args.step_size}th epoch")
    model = FaceNetModel(embedding_size=args.embedding_size, num_classes=args.num_classes, pretrained=pretrain).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    triplet_loss = TripletLoss(args.margin).to(device)

    if fc_only:
        model.freeze_all()
        model.unfreeze_fc()
        model.unfreeze_classifier()
    if except_fc:
        model.unfreeze_all()
        model.freeze_fc()
        model.freeze_classifier()
    if train_all:
        model.unfreeze_all()

    if args.load_best:
        checkpoint = './log/best_state.pth'
        print('loading', checkpoint)
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.step(checkpoint['epoch'])
        print(f"Last epoch: {checkpoint['epoch']}"
              f"Last accuracy: {checkpoint['accuracy']}"
              f"Last loss: {checkpoint['loss']}")


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

            pos_cls = batch_sample['pos_class'].to(device)
            neg_cls = batch_sample['neg_class'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                # choose the hard negatives only for "training"
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

                pos_hard_cls = pos_cls[hard_triplets]
                neg_hard_cls = neg_cls[hard_triplets]

                anc_img_pred = model.module.forward_classifier(anc_hard_img)
                pos_img_pred = model.module.forward_classifier(pos_hard_img)
                neg_img_pred = model.module.forward_classifier(neg_hard_img)

                triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()

                dists = l2_dist.forward(anc_embed, pos_embed)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.ones(dists.size(0)))

                dists = l2_dist.forward(anc_embed, neg_embed)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.zeros(dists.size(0)))

                triplet_loss_sum += triplet_loss.item()

        avg_triplet_loss = triplet_loss_sum / data_size[phase]
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

        write_csv(f'log/{phase}.csv', [epoch, np.mean(accuracy), avg_triplet_loss])

        if phase == 'valid':
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
