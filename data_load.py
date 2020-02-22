class LRW_Dataset_AV():

    def __init__(self, folds,
                 labels_file = './data/label_sorted.txt', 
                 data_dir = '/beegfs/cy1355/lipread_datachunk/', 
                 transform = None):
        """
        Args:
            labels_file (string): Path to the text file with labels
            data_dir (string): Path to the file with the facial landmarks and audio features (MFCC)
            folds (string): train / val / test indicator
            transform (callable, optional): Optional transform to be applied
                on a sample
        """
        self.folds = folds
        self.labels_file = labels_file
        with open(self.labels_file) as myfile:
            self.data_dir = myfile.read().splitlines()
        self.video_files = []
        self.audio_files = []
            
        self.v_list = {}
        self.a_list = {}
        
        for d in self.data_dir:
            self.video_files += (glob.glob(os.path.join(d, self.folds, 'video.npy')))
            self.audio_files += (glob.glob(os.path.join(d, self.folds, 'audio.npy')))
  
        for i, x in enumerate(self.video_files):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.v_list[i] = [x]
                    self.v_list[i].append(j)
        
        print('Load {} part'.format(self.folds))

    def __len__(self):
        return "length placeholder"

    def __getitem__(self, idx):
            
        if self.folds == 'test':
            video = npy_loader_aug_test(self.v_list[idx][0])
            audio = npy_loader_aug_test(self.a_list[idx][0])
            labels = self.v_list[idx][1]
            return (video, audio), labels
        else:
            video = npy_loader_aug(self.v_list[idx][0])
            audio = npy_loader_aug(self.a_list[idx][0])
            labels = self.v_list[idx][1]
            return (video, audio), labels

        return sample

def npy_loader_aug(file, v_flag):
    
    data = np.load(file)
    if v_flag == 1:
        keypoints = torch.tensor(data).float()

        keypoints_move = keypoints * 0.7
        ones = torch.ones(keypoints.shape, dtype = torch.float)
        randint = torch.randint(1,73,(1,),dtype = torch.float)
        d = keypoints_move + ones * randint
    
    else:
        d = torch.tensor(data).float()
    return d

def npy_loader_aug_test(file, v_flag):
    
    data = np.load(file)
    if v_flag == 1:
        keypoints = torch.tensor(data).float()

        keypoints_move = keypoints * 0.7
        ones = torch.ones(keypoints.shape, dtype = torch.float)
        randint = torch.randint(1,73,(1,),dtype = torch.float)
        d = keypoints_move + ones * 38
    
    else:
        d = torch.tensor(data).float()
    return d




