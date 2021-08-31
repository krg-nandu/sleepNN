class Config():
    def __init__(self):
        # experiment type
        self.exp = ['RSC', 'PFC'][0] 
        
        self.data_path = 'data/'
        self.experiments = ['RSC_LFP_rat3_500Hz.mat'] #, 'RSC_LFP_rat4_600Hz.mat','RSC_LFP_rat5_600Hz.mat']
        self.test_experiment = 'RSC_LFP_rat3_500Hz.mat'

        #self.experiments = ['PFC_LFP_rat1.mat'] #, 'PFC_LFP_rat2.mat']
        #self.test_experiment = ['PFC_LFP_rat1.mat']

        self.save_path = 'ckpts/'

        # training related hyperparameters
        self.dataset_size = 1000000
        self.n_timesteps = 32 #256 #32
        self.train_params = {
                'batch_size': 1024,
                'shuffle': False,
                'num_workers': 4
                }
        
        self.max_epochs = 100
        self.train_zscore = False
        self.model_name = 'rodentRSC'
        self.per_timestep_supervision = True

        # the mode of model training
        self.model_type = ['awake', 'asleep'][1]
        
        # the kind of training data we provide to the models
        self.style = ['raw', 'pca'][1] 

        # model related hyperparameters
        self.embedding_dim = 3
        self.hidden_dim = 128
        self.output_dim = 3

        # params for LFP data preprocessing
        self.order = 6
        self.fs = 600.
        self.cutoff = 6.
        self.window = 5 * self.fs
        self.min_bout_duration = 1. * self.fs
        self.merge_thresh = 2. * self.fs
        self.pca_window = (1/6)*self.fs
