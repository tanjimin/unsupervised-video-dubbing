class LRW_Dataset_AV(Dataset):
    """Face Landmarks and audio dataset (pre-processed data from LRW)."""

    def __init__(self, labels_file = './data/label_sorted.txt', 
                 data_dir = '/beegfs/cy1355/lipread_datachunk/', 
                 folds,
                 transform = None):
        """
        Args:
            labels_file (string): Path to the text file with labels
            data_dir (string): Path to the file with the facial landmarks and audio features (MFCC)
            folds (string): train / val / test indicator
            transform (callable, optional): Optional transform to be applied
                on a sample
        """
      
        self.labels_file = labels_file
        with open(self.labels_file) as myfile:
            self.data_dir = myfile.read().splitlines()

        self.video_files_path = os.path.join(self.data_dir, '|', self.folds, 'video.npy')
        self.audio_files_path = os.path.join(self.data_dir, '|', self.folds, 'audio.npy')
        self.video_files = []
        self.audio_files = []
        
        for d in self.data_dir:
            self.video_files += (glob.glob(os.path.join(d, self.folds, 'video.npy')))
            self.audio_files += (glob.glob(os.path.join(d, self.folds, 'audio.npy')))
  
        for i, x in enumerate(self.data_files):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)
        
        print('Load {} part'.format(self.folds))

    def __len__(self):
        return "length placeholder"

    def __getitem__(self, idx):
            
        if self.folds == 'test':
            video, audio = npz_loader_aug_test(self.list[idx][0])        
            labels = self.list[idx][1]
            return (video, audio), labels
        else:
            video, audio = npz_loader_aug(self.list[idx][0])        
            labels = self.list[idx][1]
            return (video, audio), labels

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
