import numpy as np
import matplotlib.pyplot as plt
from parameters import *


class Stimulus:

    def __init__(self):

        # generate tuning functions
        self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions()


    def generate_trial(self, test_mode = False, set_rule = None):

        if par['trial_type'] in ['DMS','DMRS45','DMRS90','DMRS90ccw','DMRS180','DMC',\
            'DMS+DMRS','DMS+DMRS_early_cue', 'DMS+DMRS_full_cue', 'DMS+DMC','DMS+DMRS+DMC','location_DMS']:
            trial_info = self.generate_basic_trial(test_mode, set_rule)
        elif par['trial_type'] in ['ABBA','ABCA']:
            trial_info = self.generate_ABBA_trial(test_mode)
        elif par['trial_type'] == 'dualDMS':
            trial_info = self.generate_dualDMS_trial(test_mode)
        elif par['trial_type'] == 'distractor':
            trial_info = self.generate_distractor_trial()

        # input activity needs to be non-negative
        trial_info['neural_input'] = np.maximum(0., trial_info['neural_input'])

        return trial_info

    def generate_dualDMS_trial(self, test_mode):

        """
        Generate a trial based on "Reactivation of latent working memories with transcranial magnetic stimulation"

        Trial outline
        1. Dead period
        2. Fixation
        3. Two sample stimuli presented
        4. Delay (cue in middle, and possibly probe later)
        5. Test stimulus (to cued modality, match or non-match)
        6. Delay (cue in middle, and possibly probe later)
        7. Test stimulus

        INPUTS:
        1. sample_time (duration of sample stimlulus)
        2. test_time
        3. delay_time
        4. cue_time (duration of rule cue, always presented halfway during delay)
        5. probe_time (usually set to one time step, always presented 3/4 through delay
        """

        test_time_rng = []
        mask_time_rng = []

        for n in range(2):
            test_time_rng.append(range((par['dead_time']+par['fix_time']+par['sample_time']+(n+1)*par['delay_time']+n*par['test_time'])//par['dt'], \
                (par['dead_time']+par['fix_time']+par['sample_time']+(n+1)*par['delay_time']+(n+1)*par['test_time'])//par['dt']))
            mask_time_rng.append(range((par['dead_time']+par['fix_time']+par['sample_time']+(n+1)*par['delay_time']+n*par['test_time'])//par['dt'], \
                (par['dead_time']+par['fix_time']+par['sample_time']+(n+1)*par['delay_time']+n*par['test_time']+par['mask_duration'])//par['dt']))


        fix_time_rng = []
        fix_time_rng.append(range(par['dead_time']//par['dt'], (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']))
        fix_time_rng.append(range((par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time'])//par['dt'], \
            (par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+par['test_time'])//par['dt']))


        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']


        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'test'            :  np.zeros((par['batch_size'],2,2),dtype=np.int8),
                      'test_mod'        :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'rule'            :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'catch'           :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'probe'           :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_size'], par['n_input']))}


        for t in range(par['batch_size']):

            # generate sample, match, rule and prob params
            for i in range(2):
                trial_info['sample'][t,i] = np.random.randint(par['num_motion_dirs'])
                trial_info['match'][t,i] = np.random.randint(2)
                trial_info['rule'][t,i] = np.random.randint(2)
                trial_info['catch'][t,i] = np.random.rand() < par['catch_trial_pct']
                if i == 1:
                    # only generate a pulse during 2nd delay epoch
                    trial_info['probe'][t,i] = np.random.rand() < par['probe_trial_pct']


            # determine test stimulu based on sample and match status
            for i in range(2):

                if test_mode:
                    trial_info['test'][t,i,0] = np.random.randint(par['num_motion_dirs'])
                    trial_info['test'][t,i,1] = np.random.randint(par['num_motion_dirs'])
                else:
                    # if trial is not a catch, the upcoming test modality (what the network should be attending to)
                    # is given by the rule cue
                    if not trial_info['catch'][t,i]:
                        trial_info['test_mod'][t,i] = trial_info['rule'][t,i]
                    else:
                        trial_info['test_mod'][t,i] = (trial_info['rule'][t,i]+1)%2

                    # cued test stimulus
                    if trial_info['match'][t,i] == 1:
                        trial_info['test'][t,i,0] = trial_info['sample'][t,trial_info['test_mod'][t,i]]
                    else:
                        sample = trial_info['sample'][t,trial_info['test_mod'][t,i]]
                        bad_directions = [sample]
                        possible_stim = np.setdiff1d(list(range(par['num_motion_dirs'])), bad_directions)
                        trial_info['test'][t,i,0] = possible_stim[np.random.randint(len(possible_stim))]

                    # non-cued test stimulus
                    trial_info['test'][t,i,1] = np.random.randint(par['num_motion_dirs'])


            """
            Calculate input neural activity based on trial params
            """
            # SAMPLE stimuli
            trial_info['neural_input'][par['sample_time_rng'], t, :] += np.reshape(self.motion_tuning[:,0,trial_info['sample'][t,0]],(1,-1))
            trial_info['neural_input'][par['sample_time_rng'], t, :] += np.reshape(self.motion_tuning[:,1,trial_info['sample'][t,1]],(1,-1))

            # Cued TEST stimuli
            trial_info['neural_input'][test_time_rng[0], t, :] += np.reshape(self.motion_tuning[:,trial_info['test_mod'][t,0],trial_info['test'][t,0,0]],(1,-1))
            trial_info['neural_input'][test_time_rng[1], t, :] += np.reshape(self.motion_tuning[:,trial_info['test_mod'][t,1],trial_info['test'][t,1,0]],(1,-1))

            # Non-cued TEST stimuli
            trial_info['neural_input'][test_time_rng[0], t, :] += np.reshape(self.motion_tuning[:,(1+trial_info['test_mod'][t,0])%2,trial_info['test'][t,0,1]],(1,-1))
            trial_info['neural_input'][test_time_rng[1], t, :] += np.reshape(self.motion_tuning[:,(1+trial_info['test_mod'][t,1])%2,trial_info['test'][t,1,1]],(1,-1))


            # FIXATION
            trial_info['neural_input'][fix_time_rng[0], t, :] += np.reshape(self.fix_tuning[:,0],(1,-1))
            trial_info['neural_input'][fix_time_rng[1], t, :] += np.reshape(self.fix_tuning[:,0],(1,-1))

            # RULE CUE
            trial_info['neural_input'][par['rule_time_rng'][0], t, :] += np.reshape(self.rule_tuning[:,trial_info['rule'][t,0]],(1,-1))
            trial_info['neural_input'][par['rule_time_rng'][1], t, :] += np.reshape(self.rule_tuning[:,trial_info['rule'][t,1]],(1,-1))

            # PROBE
            # increase reponse of all stim tuned neurons by 10
            """
            if trial_info['probe'][t,0]:
                trial_info['neural_input'][:est,probe_time1,t] += 10
            if trial_info['probe'][t,1]:
                trial_info['neural_input'][:est,probe_time2,t] += 10
            """

            """
            Desired outputs
            """
            # FIXATION
            trial_info['desired_output'][fix_time_rng[0], t, 0] = 1
            trial_info['desired_output'][fix_time_rng[1], t, 0] = 1
            # TEST 1
            trial_info['train_mask'][ test_time_rng[0], t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed
            if trial_info['match'][t,0] == 1:
                trial_info['desired_output'][test_time_rng[0], t, 2] = 1
            else:
                trial_info['desired_output'][test_time_rng[0], t, 1] = 1
            # TEST 2
            trial_info['train_mask'][ test_time_rng[1], t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed
            if trial_info['match'][t,1] == 1:
                trial_info['desired_output'][test_time_rng[1], t, 2] = 1
            else:
                trial_info['desired_output'][test_time_rng[1], t, 1] = 1

            # set to mask equal to zero during the dead time, and during the first times of test stimuli
            trial_info['train_mask'][:par['dead_time']//par['dt'], t] = 0
            trial_info['train_mask'][mask_time_rng[0], t] = 0
            trial_info['train_mask'][mask_time_rng[1], t] = 0

        return trial_info

    def generate_distractor_trial(self):

        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']
        num_time_steps = (par['dead_time']+par['fix_time']+par['sample_time']+par['distractor_time']+par['test_time']+2*par['delay_time'])//par['dt']

        trial_info = {'desired_output'  :  np.zeros((par['n_output'], num_time_steps, par['batch_size']),dtype=np.float32),
                      'train_mask'      :  np.ones((num_time_steps, par['batch_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_size']),dtype=np.int8),
                      'distractor'      :  np.zeros((par['batch_size']),dtype=np.int8),
                      'rule'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'test'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['n_input'], num_time_steps, par['batch_size']))}

        # set to mask equal to zero during the dead time

        # end of trial epochs

        distrator_time_rng = range((par['dead_time']+par['fix_time'] + par['sample_time'] + par['delay_time'] )//par['dt'],\
            (par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['distractor_time'])//par['dt'])
        test_onset = (par['dead_time']+par['fix_time'] + par['distractor_time'] + par['sample_time'] + 2*par['delay_time'])//par['dt']

        trial_info['train_mask'][:par['dead_time']//par['dt'], :] = 0

        for t in range(par['batch_size']):

            """
            Generate trial paramaters
            """
            sample_dir = np.random.randint(par['num_motion_dirs'])
            distractor_dir = np.random.randint(par['num_motion_dirs'])

            trial_info['neural_input'][:, par['sample_time_rng'], t] += np.reshape(self.motion_tuning[:, 0, sample_dir],(-1,1))
            trial_info['neural_input'][:, distrator_time_rng, t] += np.reshape(self.motion_tuning[:, 0, distractor_dir],(-1,1))
            trial_info['neural_input'][:, :test_onset, t] += np.reshape(self.fix_tuning[:, 0],(-1,1))

            """
            Determine the desired network output response
            """
            trial_info['desired_output'][0, :test_onset, t] = 1
            trial_info['desired_output'][1+sample_dir, test_onset:, t] = 1
            trial_info['train_mask'][test_onset:test_onset+mask_duration, t] = 0
            trial_info['train_mask'][test_onset:, t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed

            """
            Append trial info
            """
            trial_info['sample'][t] = sample_dir
            trial_info['distractor'][t] = distractor_dir



        return trial_info


    def generate_basic_trial(self, test_mode, set_rule = None):

        """
        Generate a delayed matching task
        Goal is to determine whether the sample stimulus, possibly manipulated by a rule, is
        identicical to a test stimulus
        Sample and test stimuli are separated by a delay
        """

        # range of variable delay, in time steps
        var_delay_max = par['variable_delay_max']//par['dt']

        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']

        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_size']),dtype=np.int8),
                      'test'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'rule'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'catch'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'probe'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_size'], par['n_input']))}


        # set to mask equal to zero during the dead time
        trial_info['train_mask'][par['dead_time_rng'], :] = 0

        for t in range(par['batch_size']):

            """
            Generate trial paramaters
            """
            sample_dir = np.random.randint(par['num_motion_dirs'])

            test_RF = np.random.choice([1,2]) if par['trial_type'] == 'location_DMS' else 0

            rule = np.random.randint(par['num_rules']) if set_rule is None else set_rule

            if par['trial_type'] == 'DMC' or (par['trial_type'] == 'DMS+DMC' and rule == 1) or (par['trial_type'] == 'DMS+DMRS+DMC' and rule == 2):
                # for DMS+DMC trial type, rule 0 will be DMS, and rule 1 will be DMC
                current_trial_DMC = True
            else:
                current_trial_DMC = False

            match = np.random.randint(2)
            catch = np.random.rand() < par['catch_trial_pct']

            """
            Generate trial paramaters, which can vary given the rule
            """
            if par['num_rules'] == 1:
                match_rotation = int(par['num_motion_dirs']*par['rotation_match']/360)
            else:
                match_rotation = int(par['num_motion_dirs']*par['rotation_match'][rule]/360)

            """
            Determine the delay time for this trial
            The total trial length is kept constant, so a shorter delay implies a longer test stimulus
            """
            if par['var_delay']:
                s = int(np.random.exponential(scale=par['variable_delay_max']/2))
                if s <= par['variable_delay_max']:
                    eod_current = eod - var_delay_max + s
                    test_onset = (par['dead_time']+par['fix_time']+par['sample_time'] + s)//par['dt']
                else:
                    catch = 1
            else:
                test_onset = (par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'])//par['dt']

            test_time_rng =  range(test_onset, par['num_time_steps'])
            fix_time_rng =  range(test_onset)
            trial_info['train_mask'][test_onset:test_onset+mask_duration, t] = 0

            """
            Generate the sample and test stimuli based on the rule
            """
            # DMC
            if not test_mode:
                if current_trial_DMC: # categorize between two equal size, contiguous zones
                    sample_cat = np.floor(sample_dir/(par['num_motion_dirs']/2))
                    if match == 1: # match trial
                        # do not use sample_dir as a match test stimulus
                        dir0 = int(sample_cat*par['num_motion_dirs']//2)
                        dir1 = int(par['num_motion_dirs']//2 + sample_cat*par['num_motion_dirs']//2)
                        possible_dirs = list(range(dir0, dir1))
                        test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
                    else:
                        test_dir = sample_cat*(par['num_motion_dirs']//2) + np.random.randint(par['num_motion_dirs']//2)
                        test_dir = np.int_((test_dir+par['num_motion_dirs']//2)%par['num_motion_dirs'])
                # DMS or DMRS
                else:
                    #import ipdb; ipdb.set_trace()
                    # just making a bit more variable in terms of possible matches
                    matching_dir = (sample_dir + match_rotation)%par['num_motion_dirs']
                    #matching_dir = (sample_dir + match_rotation + np.random.randint(-1,2))%par['num_motion_dirs']
                    if match == 1: # match trial
                        test_dir = matching_dir
                    else:
                        possible_dirs = np.setdiff1d(list(range(par['num_motion_dirs'])), matching_dir)
                        #possible_dirs = np.setdiff1d(list(range(par['num_motion_dirs'])), np.array([(sample_dir+match_rotation+x)%par['num_motion_dirs'] for x in [-1,0,1]]))
                        test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
            else:
                #import ipdb; ipdb.set_trace() 
                test_dir = np.random.randint(par['num_motion_dirs'])
                # this next part only working for DMS, DMRS tasks
                matching_dir = (sample_dir + match_rotation)%par['num_motion_dirs']
                match = 1 if test_dir == matching_dir else 0

            """
            Calculate neural input based on sample, tests, fixation, rule, and probe
            """
            # SAMPLE stimulus
            trial_info['neural_input'][par['sample_time_rng'], t, :] += np.reshape(self.motion_tuning[:, 0, sample_dir],(1,-1))

            # TEST stimulus
            if not catch:
                trial_info['neural_input'][test_time_rng, t, :] += np.reshape(self.motion_tuning[:, test_RF, test_dir],(1,-1))

            # FIXATION cue
            if par['num_fix_tuned'] > 0:
                trial_info['neural_input'][fix_time_rng, t] += np.reshape(self.fix_tuning[:,0],(-1,1))

            # RULE CUE
            if par['num_rules']> 1 and par['num_rule_tuned'] > 0:
                trial_info['neural_input'][par['rule_time_rng'][0], t, :] += np.reshape(self.rule_tuning[:,rule],(1,-1))

            """
            Determine the desired network output response
            """
            trial_info['desired_output'][fix_time_rng, t, 0] = 1.
            if not catch:
                trial_info['train_mask'][ test_time_rng, t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed
                if match == 0:
                    trial_info['desired_output'][test_time_rng, t, 1] = 1.
                else:
                    trial_info['desired_output'][test_time_rng, t, 2] = 1.
            else:
                trial_info['desired_output'][test_time_rng, t, 0] = 1.

            """
            Append trial info
            """
            trial_info['sample'][t] = sample_dir
            trial_info['test'][t] = test_dir
            trial_info['rule'][t] = rule
            trial_info['catch'][t] = catch
            trial_info['match'][t] = match

        return trial_info


    def generate_ABBA_trial(self, test_mode):

        """
        Generate ABBA trials
        Sample stimulis is followed by up to max_num_tests test stimuli
        Goal is to to indicate when a test stimulus matches the sample
        """


        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']
        # only one receptive field in this task
        RF = 0

        trial_length = par['num_time_steps']
        ABBA_delay = par['ABBA_delay']//par['dt']
        eos = (par['dead_time']+par['fix_time']+par['ABBA_delay'])//par['dt']
        test_time_rng = []
        mask_time_rng = []
        for n in range(par['max_num_tests']):
            test_time_rng.append(range(eos+ABBA_delay*(2*n+1), eos+ABBA_delay*(2*n+2)))
            mask_time_rng.append(range(eos+ABBA_delay*(2*n+1), eos+ABBA_delay*(2*n+1) + mask_duration))


        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_size']),dtype=np.float32),
                      'test'            :  -1*np.ones((par['batch_size'],par['max_num_tests']),dtype=np.float32),
                      'rule'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_size'],par['max_num_tests']),dtype=np.int8),
                      'catch'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'probe'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'num_test_stim'   :  np.zeros((par['batch_size']),dtype=np.int8),
                      'repeat_test_stim':  np.zeros((par['batch_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_size'], par['n_input']))}


        # set to mask equal to zero during the dead time
        trial_info['train_mask'][par['dead_time_rng'], :] = 0

        # set fixation equal to 1 for all times; will then change
        trial_info['desired_output'][:, :, 0] = 1

        for t in range(par['batch_size']):

            # generate trial params
            sample_dir = np.random.randint(par['num_motion_dirs'])

            """
            Generate up to max_num_tests test stimuli
            Sequential test stimuli are identical with probability repeat_pct
            """
            stim_dirs = [sample_dir]
            test_stim_code = 0

            if test_mode:
                # used to analyze how sample and test neuronal and synaptic tuning relate
                # not used to evaluate task accuracy
                while len(stim_dirs) <= par['max_num_tests']:
                    q = np.random.randint(par['num_motion_dirs'])
                    stim_dirs.append(q)
            else:
                while len(stim_dirs) <= par['max_num_tests']:
                    if np.random.rand() < par['match_test_prob']:
                        stim_dirs.append(sample_dir)
                    else:
                        if len(stim_dirs) > 1  and np.random.rand() < par['repeat_pct']:
                            #repeat last stimulus
                            stim_dirs.append(stim_dirs[-1])
                            trial_info['repeat_test_stim'][t] = 1
                        else:
                            possible_dirs = np.setdiff1d(list(range(par['num_motion_dirs'])), [stim_dirs])
                            distractor_dir = possible_dirs[np.random.randint(len(possible_dirs))]
                            stim_dirs.append(distractor_dir)

            trial_info['num_test_stim'][t] = len(stim_dirs)

            """
            Calculate input neural activity based on trial params
            """
            # SAMPLE stimuli
            trial_info['neural_input'][par['sample_time_rng'], t, :] += np.reshape(self.motion_tuning[:, RF, sample_dir],(1,-1))

            # TEST stimuli
            # first element of stim_dirs is the original sample stimulus
            for i, stim_dir in enumerate(stim_dirs[1:]):
                trial_info['test'][t,i] = stim_dir
                #test_time_rng = range(eos+(2*i+1)*ABBA_delay, eos+(2*i+2)*ABBA_delay)
                trial_info['neural_input'][test_time_rng[i], t, :] += np.reshape(self.motion_tuning[:, RF, stim_dir],(1,-1))
                trial_info['train_mask'][mask_time_rng[i], t] = 0
                trial_info['desired_output'][test_time_rng[i], t, 0] = 0
                trial_info['train_mask'][ test_time_rng[i], t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed
                if stim_dir == sample_dir:
                    trial_info['desired_output'][test_time_rng[i], t, 2] = 1
                    trial_info['match'][t,i] = 1
                else:
                    trial_info['desired_output'][test_time_rng[i], t, 1] = 1

            trial_info['sample'][t] = sample_dir


        return trial_info


    def create_tuning_functions(self):

        """
        Generate tuning functions for the Postle task
        """

        motion_tuning = np.zeros((par['n_input'], par['num_receptive_fields'], par['num_motion_dirs']))
        fix_tuning = np.zeros((par['n_input'], 1))
        rule_tuning = np.zeros((par['n_input'], par['num_rules']))


        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/(par['num_motion_tuned']//par['num_receptive_fields'])))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/par['num_motion_dirs']))

        for n in range(par['num_motion_tuned']//par['num_receptive_fields']):
            for i in range(par['num_motion_dirs']):
                for r in range(par['num_receptive_fields']):
                    if par['trial_type'] == 'distractor':
                        if n%par['num_motion_dirs'] == i:
                            motion_tuning[n,0,i] = par['tuning_height']
                    else:
                        d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                        n_ind = n+r*par['num_motion_tuned']//par['num_receptive_fields']
                        motion_tuning[n_ind,r,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])

        for n in range(par['num_fix_tuned']):
            fix_tuning[par['num_motion_tuned']+n,0] = par['tuning_height']

        for n in range(par['num_rule_tuned']):
            for i in range(par['num_rules']):
                if n%par['num_rules'] == i:
                    rule_tuning[par['num_motion_tuned']+par['num_fix_tuned']+n,i] = par['tuning_height']*par['rule_cue_multiplier']


        return motion_tuning, fix_tuning, rule_tuning


    def plot_neural_input(self, trial_info):

        print(trial_info['desired_output'][ :, 0, :].T)
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(1, 1, 1)
        t = np.arange(0,400+500+2000,par['dt'])
        t -= 900
        t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
        #im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
        im = ax.imshow(trial_info['neural_input'][:,:,0], aspect='auto', interpolation='none')
        #plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-500,0,500,1500])
        ax.set_yticks([0, 9, 18, 27])
        ax.set_yticklabels([0,90,180,270])
        f.colorbar(im,orientation='vertical')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Motion direction')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.set_title('Motion input')
        plt.show()
        plt.savefig('stimulus.pdf', format='pdf')
