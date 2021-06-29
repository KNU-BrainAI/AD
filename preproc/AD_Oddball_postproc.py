

"""

AD post-processing for Visual Oddball data

Sangtae Ahn (stahn@knu.ac.kr)


first written 12/23/2020


"""



import mne
import os, fnmatch
from matplotlib import pyplot as plt


# data load

dPath = "D:\Matlab\Data\AD\preproc\Oddball"

files=fnmatch.filter(os.listdir(dPath),'*_p.set')

os.chdir(dPath)

# %% plot ERPs

for i in files:    
    raw = mne.io.read_raw_eeglab(i)    
    # print(raw.annotations)
    events, event_id = mne.events_from_annotations(raw)
    print(raw.info)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    
    # Read epochs
    tmin = -0.2 
    tmax = 0.6
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks, baseline=(-0.2, 0), preload=True, verbose=False)
    evokeds = [epochs[name].average() for name in ('dev', 'std')]
    colors = 'blue', 'red'
    title = 'Oddball data (dev vs. std)'
    mne.viz.plot_evoked_topo(evokeds, color=colors, title=title, background_color='w')
    plt.show()
    
    
    