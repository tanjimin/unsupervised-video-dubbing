class LRW_Dataset_AV(Dataset):
    """Face Landmarks and audio dataset (pre-processed data from LRW)."""

    def __init__(self, labels_file, data_dir = '', transform = None):
        """
        Args:
            labels_file (string): Path to the text file with labels.
            data_dir (string): Path to the file with the facial landmarks and audio features (MFCC).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
      
        self.labels_file = labels_file
        with open(self.labels_file) as myfile:
            self.data_dir = myfile.read().splitlines()

        self.data_files_path = os.path.join(self.data_dir, '/', self.folds, '*.npy')
        self.data_files = []
        for category in self.data_dir:
            self.data_files += (glob.glob(self.data_files_path.replace('|', category)))
        self.list = {}
        
        for i, x in enumerate(self.data_files):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)

        print('Load {} part'.format(self.folds))

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = np.load(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample



   class LRW():
    def __init__(self, folds, path):

        self.folds = folds  # ['train', 'val', 'test']
        self.path = path
        self.istrain = (folds == 'train')
        self.test_case = False
        
        with open('./data/label_sorted.txt') as myfile:
            self.data_dir = myfile.read().splitlines()

        self.data_files_path = os.path.join(self.path, '|', self.folds, '*.npz')
        self.data_files = []
        for category in self.data_dir:
            self.data_files += (glob.glob(self.data_files_path.replace('|', category)))
        self.list = {}
        
        for i, x in enumerate(self.data_files):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)

        print('Load {} part'.format(self.folds))

    def __getitem__(self, idx):

        if self.test_case:
            keypoints, mfcc = npz_loader_aug_test(self.list[idx][0])        
            labels = self.list[idx][1]
            return (keypoints, mfcc), labels
        else:
            keypoints, mfcc = npz_loader_aug(self.list[idx][0])        
            labels = self.list[idx][1]
            return (keypoints, mfcc), labels

    def __len__(self):
        return len(self.data_files)
