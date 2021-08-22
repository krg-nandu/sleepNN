class Config():
    def __init__(self):
        # experiment type
        self.exp = 'RSC' # [choice between RSC and PFC]
        self.data_path = 'data/'
        self.experiments = ['RSC_LFP_rat3_600Hz.mat', 'RSC_LFP_rat4_600Hz.mat','RSC_LFP_rat5_600Hz.mat']

        self.save_path = 'ckpts/'

        # training related hyperparameters
        self.dataset_size = 1000000
        self.n_timesteps = 256
        self.train_params = {
                'batch_size': 4096,
                'shuffle': False,
                'num_workers': 1
                }
        self.max_epochs = 100
        self.train_zscore = False
        self.model_name = 'rodentRSC'
        self.model_type = 'asleep' # choice is between asleep vs awake

        # params for the neural data preprocessing
        self.order = 6
        self.fs = 600.
        self.cutoff = 6.
        self.window = 5 * self.fs
        self.min_bout_duration = 1. * self.fs
        self.merge_thresh = 2. * self.fs
