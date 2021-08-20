class Config():
    def __init__(self):
        # experiment type
        self.exp = 'RSC' # [choice between RSC and PFC]

        # training related hyperparameters
        self.dataset_size = 300000
        self.n_timesteps = 32
        
