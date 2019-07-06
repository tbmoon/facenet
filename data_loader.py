import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, transform=None):

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets)

    @staticmethod
    def generate_triplets(df, num_triplets):

        def make_dictionary_for_face_class(df):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append((df.iloc[idx]['id'], df.iloc[idx]['ext']))
            return face_classes

        triplets = []
        classes = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)

        for _ in range(num_triplets):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            anc_id = face_classes[pos_class][ianc][0]
            anc_ext = face_classes[pos_class][ianc][1]
            pos_id = face_classes[pos_class][ipos][0]
            pos_ext = face_classes[pos_class][ipos][1]
            neg_id = face_classes[neg_class][ineg][0]
            neg_ext = face_classes[neg_class][ineg][1]

            triplets.append(
                [anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, anc_ext, pos_ext, neg_ext])

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, anc_ext, pos_ext, neg_ext = \
            self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, str(pos_name), str(anc_id) + f'.{anc_ext}')
        pos_img = os.path.join(self.root_dir, str(pos_name), str(pos_id) + f'.{pos_ext}')
        neg_img = os.path.join(self.root_dir, str(neg_name), str(neg_id) + f'.{neg_ext}')

        anc_img = io.imread(anc_img)
        pos_img = io.imread(pos_img)
        neg_img = io.imread(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


def get_dataloader(train_root_dir, valid_root_dir,
                   train_csv_name, valid_csv_name,
                   num_train_triplets, num_valid_triplets,
                   batch_size, num_workers):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])}

    face_dataset = {
        'train': TripletFaceDataset(root_dir=train_root_dir,
                                    csv_name=train_csv_name,
                                    num_triplets=num_train_triplets,
                                    transform=data_transforms['train']),
        'valid': TripletFaceDataset(root_dir=valid_root_dir,
                                    csv_name=valid_csv_name,
                                    num_triplets=num_valid_triplets,
                                    transform=data_transforms['valid'])}

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x in ['train', 'valid']}

    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size
