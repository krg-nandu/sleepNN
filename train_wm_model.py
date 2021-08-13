from parameters import par
import stimulus

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drnn import dRNN
import matplotlib.pyplot as plt
from matplotlib import animation

def main(device='cuda:0', gpu_id = None):
    max_batches = 1000

    #Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()
    model = dRNN(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr = par['learning_rate'])

    for n_batch in range(max_batches):
        trial_info = stim.generate_trial(set_rule = None)
        trial = torch.tensor(trial_info['neural_input']).to(device)
        
        h, y, x, u = model(trial)
        preds = torch.stack(y)
        labels = torch.tensor(trial_info['desired_output']).to(device)

        loss = F.binary_cross_entropy(preds, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('Batch:{}, Loss:{:.4f}'.format(n_batch+1, float(loss)))

    torch.save(model.state_dict(), 'ckpts/wm_v0.pth')

def make_trial_animation(stimulus, preds):

    trial_idx = np.random.randint(stimulus.shape[1])
    output_map = {0: 'Fixation', 1: 'Non-Match', 2: 'Match'}

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_values = np.arange(stimulus.shape[-1])
    line, = ax.plot(x_values, np.zeros_like(x_values))
    ax.set_ylim((stimulus.min(),stimulus.max() ))

    def animate(k):
        line.set_data(x_values, stimulus[k, trial_idx, :])
        ax.axis('off')
        ax.set_title('model output: {}'.format(output_map[preds[k, trial_idx]]))
        return (line,)

    anim = animation.FuncAnimation(fig, animate,
                               frames=stimulus.shape[0], interval=20, blit=True)
    anim.save('gifs/trial_{}.gif'.format(trial_idx), writer='imagemagick', fps=20)

def model_eval(device='cuda:0', gpu_id=None):
    stim = stimulus.Stimulus()
    model = dRNN(device)
    model = model.to(device)

    model.load_state_dict(torch.load('ckpts/wm_v0.pth'))
    model = model.eval()

    trial_info = stim.generate_trial(set_rule = None)
    test_inp = trial_info['neural_input']
    test_trial = torch.tensor(test_inp).to(device)

    h, y, x, u = model(test_trial)

    preds = torch.argmax(torch.stack(y), dim=-1)
    labels = torch.tensor(trial_info['desired_output'])

    for k in range(10):
        make_trial_animation(test_inp, preds.cpu().detach().numpy())

if __name__ == '__main__':
    #main()
    model_eval()
