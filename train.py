import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.modules.distance import PairwiseDistance
import torchvision
from torchvision import transforms
from eval_metrics import evaluate, plot_roc
from utils import TripletLoss
from models import FaceNetModel
from data_loader import TripletFaceDataset, get_dataloader


parser = argparse.ArgumentParser(description = 'Face Recognition using Triplet Loss')

parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'SE',
                    help = 'start epoch (default: 0)')
parser.add_argument('--num-epochs', default = 200, type = int, metavar = 'NE',
                    help = 'number of epochs to train (default: 200)')
parser.add_argument('--num-classes', default = 10000, type = int, metavar = 'NC',
                    help = 'number of clases (default: 10000)')
parser.add_argument('--num-train-triplets', default = 10000, type = int, metavar = 'NTT',
                    help = 'number of triplets for training (default: 10000)')
parser.add_argument('--num-valid-triplets', default = 10000, type = int, metavar = 'NVT',
                    help = 'number of triplets for vaidation (default: 10000)')
parser.add_argument('--embedding-size', default = 128, type = int, metavar = 'ES',
                    help = 'embedding size (default: 128)')
parser.add_argument('--batch-size', default = 64, type = int, metavar = 'BS',
                    help = 'batch size (default: 128)')
parser.add_argument('--num-workers', default = 8, type = int, metavar = 'NW',
                    help = 'number of workers (default: 8)')
parser.add_argument('--learning-rate', default = 0.001, type = float, metavar = 'LR',
                    help = 'learning rate (default: 0.001)')
parser.add_argument('--margin', default = 0.5, type = float, metavar = 'MG',
                    help = 'margin (default: 0.5)')
parser.add_argument('--train-root-dir', default = '/run/media/hoosiki/WareHouse2/home/mtb/datasets/vggface2/test_mtcnnpy_182', type = str,
                    help = 'path to train root dir')
parser.add_argument('--valid-root-dir', default = '/run/media/hoosiki/WareHouse2/home/mtb/datasets/lfw/lfw_mtcnnpy_182', type = str,
                    help = 'path to valid root dir')
parser.add_argument('--train-csv-name', default = './datasets/test_vggface2.csv', type = str,
                    help = 'list of training images')
parser.add_argument('--valid-csv-name', default = './datasets/lfw.csv', type = str,
                    help = 'list of validtion images')

args    = parser.parse_args()
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)


def main():
    
    model     = FaceNetModel(embedding_size = args.embedding_size, num_classes = args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)
    
    if args.start_epoch != 0:
        checkpoint = torch.load('./log/checkpoint_epoch{}.pth'.format(args.start_epoch-1))
        model.load_state_dict(checkpoint['state_dict'])
    
    for epoch in range(args.start_epoch, args.num_epochs + args.start_epoch):
        
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, args.num_epochs + args.start_epoch - 1))
    
        data_loaders, data_size = get_dataloader(args.train_root_dir,     args.valid_root_dir,
                                                 args.train_csv_name,     args.valid_csv_name,
                                                 args.num_train_triplets, args.num_valid_triplets,   
                                                 args.batch_size,         args.num_workers)

        train_valid(model, optimizer, scheduler, epoch, data_loaders, data_size)

    print(80 * '=')

    
def train_valid(model, optimizer, scheduler, epoch, dataloaders, data_size):
    
    for phase in ['train', 'valid']:

        labels, distances = [], []
        triplet_loss_sum  = 0.0

        if phase == 'train':
            scheduler.step()
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
                
                anc_hard_embed = anc_embed[hard_triplets].to(device)
                pos_hard_embed = pos_embed[hard_triplets].to(device)
                neg_hard_embed = neg_embed[hard_triplets].to(device)
            
                anc_hard_img   = anc_img[hard_triplets].to(device)
                pos_hard_img   = pos_img[hard_triplets].to(device)
                neg_hard_img   = neg_img[hard_triplets].to(device)
            
                pos_hard_cls   = pos_cls[hard_triplets].to(device)
                neg_hard_cls   = neg_cls[hard_triplets].to(device)
            
                anc_img_pred   = model.forward_classifier(anc_hard_img).to(device)
                pos_img_pred   = model.forward_classifier(pos_hard_img).to(device)
                neg_img_pred   = model.forward_classifier(neg_hard_img).to(device)
            
                triplet_loss   = TripletLoss(args.margin).forward(anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)
        
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
        labels           = np.array([sublabel for label in labels for sublabel in label])
        distances        = np.array([subdist for dist in distances for subdist in dist])
    
        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))
   
        with open('./log/{}_log_epoch{}.txt'.format(phase, epoch), 'w') as f:
            f.write(str(epoch)             + '\t' +
                    str(np.mean(accuracy)) + '\t' +
                    str(avg_triplet_loss))
            
        if phase == 'train':
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                        './log/checkpoint_epoch{}.pth'.format(epoch))
        else:
            plot_roc(fpr, tpr, figure_name = './log/roc_valid_epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    
    main()
