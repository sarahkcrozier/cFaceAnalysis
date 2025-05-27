"""
Analyse cface1 face data
Mainly using the out.csv log file, and the OpenFace-processed .csv file with action unit time series
Within each trial, key events are 'trigger', 'stimMove', 'stimStop', 'fixation'
Resample action unit series from being indexed by frames, to be indexed by time (sec)

Get the following metrics of facial movement for each subject:
1) amplitudes: Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12)). Does this decrease with time due to tiredness?
2) cent_ts: central tendency time series from trigger to next Instruct, averaged across trials, when they're asked to smile (HA) or frown(AN) separately. Get this for each action unit. Could use these as fMRI regressor
3) cent_ts_pca: Point 3 but for first principal component
4) latencies: Distribution of latency post-trigger for the smile or frown action. Look at distribution and exclude outliers
5) maxgrad: Maximum value of first derivative for AU12 or PCA component

Deal with amplitude as confounder

Issues:
- Many trials have no response. No sudden uptick. So average time series may not be accurate
- Even trials with a response are not well modelled by average time series, which is a smooth curve. This is because the average time series is a smooth curve, whereas the actual time series is more like a step function.


new columns:
exclude_p05: boolean, whether to exclude this subject based on equalizing group05's age distribution (p>0.05)
exclude_p5: boolean, as above, (p>0.5)

"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, pingouin as pg
import acommonfuncs, acface_utils
from acface_utils import ntrials, n_trialsperemotion, emots, relevant_timejitters, relevant_labels, relevant_timestamps, midpoint_timestamps
from acommonvars import *
from scipy.interpolate import interp1d
c = acommonfuncs.clock()

plt.rcParams.update({'font.size': 13})

### SETTABLE PARAMETERS ###
group='group05' #default 'group05
print(f'Group: {group}')
static_or_dynamic = 'static' #whether au_static was used in OpenFace execution or not
action_unit = 'AU12' #which action unit to plot
min_success = 0.90 #minimum proportion of successful frames for a subject to be included, default 0.9
min_latency_validperc = 70 #minimum proportion of trials with valid latency for a subject to be included, default 80
max_response_duration = 1.5 #maximum duration of response in seconds (excludes subject 028 for slow response), default 1.5
target_fps=acface_utils.target_fps
times_trial_regular = acface_utils.times_trial_regular

t['use_cface'] = ((include) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #those subjects whose cface data we will use

ha_AU='AU12'
ha_AU_index = aus_labels.index(ha_AU)

get_amplitudes=True
get_cent_ts=True
get_cent_ts_pca=True 
get_latencies=True 
get_maxgrads=True
to_plot=True #plots for each participant
load_table=True
save_table=False
age_matching = True #whether to exclude subjects to equalize age distribution across groups
if age_matching: print("AGE-MATCHED GROUPS")

bad_mri_subjects = [108,75,34,66,20,26,84]

central_tendency = np.median #this function can be np.mean or np.median
print(f'Using {central_tendency} as central tendency')

gps = [i for i in t[group].unique() if i!='']
if gps[0]=='hc': #make sure healthy controls are not first in the list
    gps = [gps[1],gps[0]]
print('gps',gps)
t['cface_bad_mri'] = t.record_id.isin(bad_mri_subjects)

#We will later be getting AU time series for each trial, and resampling timestamps to be relative to trigger as 0, and at 20fps (target_fps). Here we get resampled per-trial timestamps


def get_outcomes(subject):
    """
    Intermediate variables: all dictionaries have two keys: 'ha' and 'an', for when pt was asked to smile or frown

    times_eachframe: timestamps for raw time series, np.array of shape n_frames (9377)
    aus: raw time series, np.array of shape n_frames(9377)*nAUs(16)
    times_regular: like timestamps, but resampled to 20Hz, np.array of shape n_frames (10243)
    aust: like aus, but resampeld to 20Hz, np.array of shape n_frames*nAUs

    times_trial_regular: resampled timestamps for a single trial, np.array of shape n_frames (88)
    aus_trial: dict, each value contains resampled time series for each trial separately, np.array of shape n_trials(40 trials per emotion)*n_frames(88)*nAUs(16)
    aus_trial_cent: dict, each value contains central tendency across trials, np.array of shape n_frames(88)*nAUs(16)

    aus_pca: dict, each value contains PCA-transformed raw time series for each emotion separately, np.array of shape n_frames(9377)*nPCs(16). The same data but transformed to smile instruction PCA components (key 'ha') or frown instruction PCA components (key 'an')
    aust_pca: like aus_pca, but contains resampled time series, np.array of shape n_frames(10243)*nPCs(16)
    aus_trial_pca: dict, each value contains PCA-transformed single-trial data for each emotion separately, np.array of shape n_trials(40)*n_frames(88)*nPCs(16)
    aus_trial_pca_cent: dict, each value contains central tendency across trials from aus_trial_pca, np.array of shape n_frames(88)*nPCs(16)
    """
    print(f'{c.time()[1]}: sub {subject}')
    all_frames,aus,success = acommonfuncs.get_openface_table('cface1',subject,static_or_dynamic) #Get the OpenFace intermediates .csv for this subject
    webcam_frames_success_proportion = np.sum(success)/len(success)
    if webcam_frames_success_proportion < min_success:
        print(f"WARNING: {subject} has only {webcam_frames_success_proportion:.3f} proportion of successful frames. Returning nan.")
        keys = ['amp_max','amp_range','cent_ts','cent_ts_pca']
        #t.loc[t.record_id==int(subject),'cface_goodwebcam']=False
        return {key:np.nan for key in keys}

    df = acommonfuncs.get_beh_data('cface1',subject,'out',use_MRI_task=False,source_data_raw=False) #Get behavioural data from *out.csv

    """Get face summary data from *face.csv"""
    face_summary = acommonfuncs.get_beh_data('cface1',subject,'face',use_MRI_task=False,source_data_raw=False)
    #face_summary=pd.read_csv(glob(f"{resultsFolder}*face.csv")[0]) # make face summary csv into dataframe 
    face_summary = {i[0]:i[1] for i in face_summary.values} #convert face summary array into dictionary
    camtstart = face_summary['camtstart'] #time when webcam started recording
    camactualfps = face_summary['camactualfps'] #actual fps of webcam recording

    """
    From out.csv log file, get all pairs of (time,framenumber). Interpolate between frames to calculate a time since onset for each frame number in webcam data. Now the action unit time series is indexed by timestamp instead of frame number. Then interpolate timestamps to resample the AU time series at regular intervals of exactly 20fps.
    """
    times,frames = acface_utils.get_all_timestamps_and_framestamps(df,ntrials)
    duplicate_indices = acface_utils.find_duplicate_indices(times) + acface_utils.find_duplicate_indices(frames)
    valid_indices = [i for i in range(len(frames)) if i not in duplicate_indices] #remove entries where same frame number corresponds to multiple timestamps, or timestamp corresponds to multiple frame numbers (usually due to dropped frames)
    interp_frametimes = interp1d(frames[valid_indices],times[valid_indices],kind='linear',fill_value = 'extrapolate')
    times_eachframe = interp_frametimes(all_frames) #get time since onset for each frame number, corresponding to rows of aus
    interp_aus = interp1d(times_eachframe, aus, axis=0, kind='linear',fill_value = 'extrapolate')
    times_regular = np.arange(0,np.max(times_eachframe),1/target_fps)
    aust = interp_aus(times_regular)
    aust = pd.DataFrame(aust)
    aust.columns=aus.columns

    """
    Get interpolated AU time series for each trial separately. Trials are grouped into 2 groups: when they're asked to smile (HA) or frown (AN). In detail: Get timestamps and AU values for each trial, from trigger to next Instruct. Set timestamps to be relative to the trigger as 0. Find a linear interpolation of values, and resample at 20 fps.
    Each variable below is a dictionary with keys 'ha' and 'an', for when pt was asked to smile or frown
    aus_trial['ha'] has interpolated AU values for each trial separately: array size n_smiletrials (40) * n_frames (80) * nAUs (16)
    """
    aus_trial={}
    for emot in emots:
        aus_trial[emot] = acface_utils.get_all_post_trigger_time_series(df,interp_aus,times_trial_regular,emotion=emot)

    if get_amplitudes:
        """Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12))"""
        aus_trial_max,aus_trial_min,aus_trial_range = {},{},{}
        for emot in emots:
            aus_trial_max[emot] = np.max(aus_trial[emot],axis=1)
            aus_trial_min[emot] = np.min(aus_trial[emot],axis=1)
            aus_trial_range[emot] = aus_trial_max[emot] - aus_trial_min[emot]
        ha_AU_trial_ha_max = aus_trial_max['ha'][:,ha_AU_index] #OUTCOME amplitudes
        ha_AU_trial_ha_range = aus_trial_range['ha'][:,ha_AU_index] #OUTCOME amplitudes
    

    if get_cent_ts:
        #Get cent (across trials) of the per-trial AU time series when they're asked to smile (HA) or frown(AN) separately. Could use these as fMRI regressor
        aus_trial_cent={} #aus_trial_cent['ha'] has cent across trials, separately for each timepoint
        for emot in emots:
            aus_trial_cent[emot]=central_tendency(aus_trial[emot],axis=0) #OUTCOME cent_ts

    if get_cent_ts_pca:
        #PCA of action unit time series for each emotion separately
        pca,comp0,aus_pca,aus_trial_pca,aust_pca,aus_trial_pca_cent={},{},{},{},{},{}
        for emot in emots:

            pca[emot] = acface_utils.get_pca(aus_trial[emot])
            if not(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_cent[emot],pca[emot])):
                pca[emot].components_[0] = -pca[emot].components_[0] #ensure component zero increases from trigger to middle of stimMove
            comp0[emot] = pca[emot].components_[0]
            aus_trial_pca[emot] = np.array([pca[emot].transform(i) for i in aus_trial[emot]]) 
            aus_trial_pca_cent[emot] = pca[emot].transform(aus_trial_cent[emot]) #OUTCOME cent_ts_pca
            aus_pca[emot] = acface_utils.pca_transform(pca[emot],aus.values) 
            aust_pca[emot] = acface_utils.pca_transform(pca[emot],aust.values) 
        #interp_aus_pca = interp1d(times_eachframe,aus_pca['ha'],axis=0,kind='linear',fill_value = 'extrapolate')  
        assert(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_cent[emot],pca[emot]))

    """
    for emot in emots:
        for ntrial in range(n_trialsperemotion):
            #aus_trial_pca[emot]
    """

    if get_latencies and get_maxgrads:
        get_latency = lambda values: acface_utils.get_latency(values,target_fps,n_trialsperemotion,times_trial_regular,plot=False) 
        other_metrics= {key:get_latency(aus_trial_pca[key][:,:,0]) for key in emots} #metrics around latency, using first PCA component of smile and frown trials
        max_grad_ha_AU12 = get_latency(aus_trial['ha'][:,:,ha_AU_index])['max_grad'] #max gradient for smile trials using AU12

    """
    For each trial, get the difference between facial response amplitude when the stimulus finishes moving and facial response amplitude when the stimulus starts moving. Get a t-statistic for the difference in this amplitude between congruent and incongruent trials (e.g. HAHA-HAAN), and return. Repeat for each instruction type (smile or frown)
    """
    #For each instruction condition, get boolean array of which trials are congruent
    congruent_trials = dict()
    for emot in emots:
        matching_rows = df.ptemot==emot.upper()
        congruent_trials[emot] = (df.cong[matching_rows]==1).values 
    """
    rng = np.random.default_rng()
    for emot in emots:
        rng.shuffle(congruent_trials[emot])
    """
    #For each trial in each instruction condition, get the normalized amplitude at end of the trial
    end_amps = {key: np.zeros(n_trialsperemotion) for key in emots}
    time_before_stimMove = np.sum(relevant_timejitters[0:3])
    time_after_stimMove = np.sum(relevant_timejitters[0:4])
    index_before_stimMove = np.where(np.isclose(times_trial_regular,time_before_stimMove))[0][0] 
    index_after_stimMove = np.where(np.isclose(times_trial_regular,time_after_stimMove))[0][0] #index of the end of stimMove in single trial time series
    #print(time_before_stimMove, time_after_stimMove)
    for emot in emots:
        for i in range(n_trialsperemotion): #check that the time series for each trial is increasing from trigger to end of stimMove (i.e. that the PCA component 0 is increasing
            time_series = aus_trial_pca[emot][i,:,0] #get time series for ith trial, PCA component 0
            #time_series = aus_trial[emot][i,:,ha_AU_index] #use AU12
            val_before_stim = time_series[index_before_stimMove]
            val_after_stim = time_series[index_after_stimMove]
            end_amp = val_after_stim - val_before_stim
            """
            #To normalize by the range of the time series
            minval = np.min(time_series[0:index_after_stimMove])
            maxval = np.max(time_series[0:index_after_stimMove])
            end_amp = (val_after_stim-minval)/(maxval-minval)
            """
            end_amps[emot][i] = end_amp

    #Calculate the t-statistic for the difference in trial end amplitude between congruent and incongruent trials
    end_amp_tstat = {}
    for emot in emots:
        amps_congruent = end_amps[emot][congruent_trials[emot]]
        amps_incongruent = end_amps[emot][~congruent_trials[emot]]
        tvalue = pg.ttest(amps_congruent,amps_incongruent)['T'].iloc[0] # t statistic for cong > incong
        if emot=='ha':
            print(f"{emot}: congruent={np.mean(amps_congruent):.3f}, incongruent={np.mean(amps_incongruent):.3f}, t={tvalue:.3f}")
        end_amp_tstat[emot] = tvalue

    #save t-statistic for congruent>incongruent, for amplitude (beforeStimMove removed), for each time point separately
    aus_pca_amp_diffXcond = {'ha':[],'an':[]} 
    for emot in emots:
        tts = aus_trial_pca[emot][:,:,0]
        tts2 = tts - np.tile(tts[:,index_before_stimMove],(len(times_trial_regular),1)).T #subtract the value at start of stimMove
        array = np.zeros(len(times_trial_regular))
        for i in range(len(times_trial_regular)):
            array[i] = pg.ttest(tts2[congruent_trials[emot],i],tts2[~congruent_trials[emot],i])['T'].iloc[0]
        aus_pca_amp_diffXcond[emot] = array


    #Save central tendency time series for each instruction condition, separately for congruent and incongruent trials
    aus_trial_pca_cond_cent = {'ha':{'cong':[],'incong':[]},'an':{'cong':[],'incong':[]}}
    for emot in emots:
        aus_trial_pca_cond_cent[emot]['cong'] = central_tendency(aus_trial_pca[emot][congruent_trials[emot],:,0], axis=0)
        aus_trial_pca_cond_cent[emot]['incong'] = central_tendency(aus_trial_pca[emot][~congruent_trials[emot],:,0], axis=0)

    """
    fig,ax=plt.subplots(2)
    ax[0].plot(times_trial_regular,aus_pca_amp_diffXcond['ha'],color='green',label='ha')
    ax[0].plot(times_trial_regular,aus_pca_amp_diffXcond['an'],color='red',label='an')
    ax[0].axvline(x=time_before_stimMove)
    ax[0].axvline(x=time_after_stimMove)
    ax[0].set_title(f"ha vs an, end_amp {end_amp_tstat['ha']:.3f}")
    plt.show(block=False)
    
    fig,ax=plt.subplots(2)
    ax[0].plot(times_trial_regular,np.mean(aus_trial_pca['ha'][congruent_trials['ha'],:,0], axis=0),color='blue',label='cong')
    ax[0].plot(times_trial_regular,np.mean(aus_trial_pca['ha'][~congruent_trials['ha'],:,0], axis=0),color='red',label='incong')
    ax[0].axvline(x=time_before_stimMove)
    ax[0].axvline(x=time_after_stimMove)
    ax[0].set_title(f"cong vs incong, end_amp {end_amp_tstat['ha']:.3f}")
    plt.show(block=False)
    assert(0)
    """

    """Plotting for single subject"""
    if to_plot:       
        plot_this_au_trial = lambda values,title,results: acface_utils.plot_this_au_trial(values,title,times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps,results=results,final_timings_only=False)

        emot='ha'
        values = aus_trial[emot][:,:,ha_AU_index]
        title = f'{ha_AU} time series for some {emot} trials'
        plot_this_au_trial(values,title,other_metrics['ha'])

        emot='ha'
        values = aus_trial_pca[emot][:,:,0]
        title = f'PCA comp 0 time series for some {emot} trials'
        plot_this_au_trial(values,title,other_metrics['ha'])

        emot='an'
        values = aus_trial_pca[emot][:,:,0]
        title = f'PCA comp 0 time series for some {emot} trials'
        plot_this_au_trial(values,title,other_metrics['an'])

        acface_utils.plot_this_au_trial_superimposed('ha',ha_AU_index,'instruction smile',ha_AU,aus_trial,aus_trial_cent,relevant_timestamps,relevant_labels,midpoint_timestamps,times_trial_regular)
        acface_utils.plot_this_au_trial_superimposed('ha',0,'instruction smile','First PC (smile)', aus_trial_pca,aus_trial_pca_cent,relevant_timestamps,relevant_labels,midpoint_timestamps,times_trial_regular)
        acface_utils.plot_this_au_trial_superimposed('an',0,'instruction frown','First PC (frown)', aus_trial_pca,aus_trial_pca_cent,relevant_timestamps,relevant_labels,midpoint_timestamps,times_trial_regular)

        fig,ax=plt.subplots(figsize=(6,3))
        #fig.set_size_inches(18,3) #18,3
        acface_utils.plot_this_au(df,ax,times_regular,aust,this_AU='AU12',color='mediumpurple',label='smile',ylabel='AU12')
        fig.tight_layout()

        fig,ax=plt.subplots(figsize=(6,3))
        #fig.set_size_inches(18,3) #18,3
        acface_utils.plot_this_au(df,ax,times_regular,aust_pca['ha'],this_AU='comp0',color='mediumpurple',label='smile',ylabel='First PC')
        acface_utils.plot_this_au(df,ax,times_regular,aust_pca['an'],this_AU='comp0',color='indianred',label='frown',ylabel='First PC')
        fig.tight_layout()

        #acface_utils.plot_all_aus(df,times_regular,aust)
        #acface_utils.plot_all_aus(df,times_regular,aust_pca['ha'])   
        #acface_utils.plot_all_aus(df,times_regular,aust_pca['an'])
        #print([[aus_names[i],comp0[i]] for i in range(len(comp0))] )

        #Plot to see whether response amplitudes are decreasing with time due to tiredness
        fig,ax = plt.subplots()
        ax.scatter(range(len(ha_AU_trial_ha_max)) , ha_AU_trial_ha_max) 
        ax.set_xlabel('Trial')
        ax.set_ylabel('Max AU12 value from trigger to end of nextInstruct')

        #acface_utils.plot_pca_mapping(pca, aus_pca,aus_names)
        plt.show(block=False)

    return {'amp_max':ha_AU_trial_ha_max, 'amp_range':ha_AU_trial_ha_range,'cent_ts': aus_trial_cent,'cent_ts_pca':aus_trial_pca_cent, 'other_metrics':other_metrics,'end_amp_tstat':end_amp_tstat,'aus_trial_pca_cond_cent':aus_trial_pca_cond_cent,'ts':aus_trial,'ts_pca':aus_trial_pca,'amp_diffXcond':aus_pca_amp_diffXcond,'max_grad_ha_AU12':max_grad_ha_AU12}

include_exclude_columns = ['use_cface','cface_outliers','cface_hasPulse','cface_goodwebcam','cface_bad_mri','cface_exclude_p05','cface_exclude_p5']
cent_ts_columns = ['cface_cent_ts_ha_pca0','cface_cent_ts_an_pca0','cface_cent_ts_ha_au12']
cong_incong_columns = [f'cface_cent_ts_{emot}_{cong_or_incong}_pca0' for emot in emots for cong_or_incong in ['cong','incong']]
more_columns = [f'cface_amp_diffXcond_{emot}' for emot in emots] 
new_columns = include_exclude_columns + cent_ts_columns +['cface_latencies_ha','cface_durations_ha','cface_latencies_an','cface_durations_an','cface_amp_max_cent','cface_amp_range_cent','cface_amp_max_slope','cface_latencies_validperc_ha','cface_latencies_validperc_an','cface_latencies_cent_ha','cface_latencies_cent_an','cface_durations_cent_ha','cface_durations_cent_an','cface_maxgrads_cent_ha','cface_maxgrads_cent_an','cface_maxgradns_cent_ha','cface_maxgradns_cent_an','cface_maxgrads_cent_ha_AU12','cface_end_amp_tstat_ha','cface_end_amp_tstat_an'] + cong_incong_columns

if load_table:
    t=acommonfuncs.add_table(t,'outcomes_cface_face.csv')
    #When csv is saved, elements which are lists are saved as strings. We need to convert them back to lists.
    t = acommonfuncs.str_columns_to_literals(t,cent_ts_columns + cong_incong_columns)

    """
    if 'cface_group05_p05' in t.columns:
        t['cface_group05_p05']=t.cface_group05_p05.fillna('') 
    if 'cface_group05_p5' in t.columns:
        t['cface_group05_p5']=t.cface_group05_p5.fillna('')
    """
        
else:
    t['cface_hasPulse'] = True
    has_no_pulse = [34]
    for subject in has_no_pulse:
        t.loc[t.record_id==subject,'cface_hasPulse'] = False
    t['cface_goodwebcam']=True
    t=acommonfuncs.add_columns(t,cent_ts_columns+cong_incong_columns+more_columns)
    t=acommonfuncs.add_columns(t,['cface_latencies_ha','cface_durations_ha','cface_latencies_an','cface_durations_an'])

    temp = np.zeros((t.shape[0],n_trialsperemotion,len(times_trial_regular)),dtype=np.float32)
    tspc0 = {'ha':temp.copy(),'an':temp.copy()} #hold all subjects' time series for principal component 0
    tsau12 = temp.copy() #as above, but only for 'ha' instruction, AU12

    outcomes = get_outcomes('020') #to visualize. 93
    assert(0)
    """
    20 good for the multiple plot (use this)
    22 not bad
    31 good overall
    """


    for t_index in range(t.shape[0]):
        if t.use_cface[t_index]:
            outcomes = get_outcomes(subs[t_index])
            #t.at[t_index,'cface_amp_max_cent'] = central_tendency(outcomes['amp_max'])
            #t.at[t_index,'cface_amp_range_cent'] = npt.co.cent(outcomes['amp_range'])
            #t.at[t_index,'cface_amp_max_slope'] = acface_utils.get_slope(outcomes['amp_max'])
            if type(outcomes['cent_ts_pca'])==dict: #exclude nans from poor webcam acquisitions
                t.at[t_index,'cface_amp_max_cent'] = central_tendency(outcomes['amp_max'])
                t.at[t_index,'cface_amp_range_cent'] = central_tendency(outcomes['amp_range'])
                t.at[t_index,'cface_amp_max_slope'] = acface_utils.get_slope(outcomes['amp_max'])

                t.at[t_index,'cface_cent_ts_ha_pca0'] = list(outcomes['cent_ts_pca']['ha'][:,0])
                t.at[t_index,'cface_cent_ts_an_pca0'] = list(outcomes['cent_ts_pca']['an'][:,0])
                t.at[t_index,'cface_cent_ts_ha_au12'] = list(outcomes['cent_ts']['ha'][:,ha_AU_index])
                r_validperc,r_latencies,r_durations,r_maxgrads,r_maxgradns=acface_utils.extract_subject_result(outcomes['other_metrics']['ha'],n_trialsperemotion)
                t.at[t_index,'cface_latencies_validperc_ha'] = r_validperc
                t.at[t_index,'cface_latencies_ha'] = r_latencies
                t.at[t_index,'cface_latencies_cent_ha'] = central_tendency(r_latencies)
                t.at[t_index,'cface_durations_ha'] = r_durations
                t.at[t_index,'cface_durations_cent_ha'] = central_tendency(r_durations)
                t.at[t_index,'cface_maxgrads_cent_ha'] = central_tendency(r_maxgrads)
                t.at[t_index,'cface_maxgradns_cent_ha'] = central_tendency(r_maxgradns)
                r_validperc,r_latencies,r_durations,r_maxgrads,r_maxgradns=acface_utils.extract_subject_result(outcomes['other_metrics']['an'],n_trialsperemotion)
                t.at[t_index,'cface_latencies_validperc_an'] = r_validperc
                t.at[t_index,'cface_latencies_an'] = r_latencies
                t.at[t_index,'cface_latencies_cent_an'] = central_tendency(r_latencies)
                t.at[t_index,'cface_durations_an'] = r_durations
                t.at[t_index,'cface_durations_cent_an'] = central_tendency(r_durations)
                t.at[t_index,'cface_maxgrads_cent_an'] = central_tendency(r_maxgrads)
                t.at[t_index,'cface_maxgradns_cent_an'] = central_tendency(r_maxgradns)
                t.at[t_index,'cface_maxgrads_cent_ha_AU12'] = central_tendency(outcomes['max_grad_ha_AU12'])
                t.at[t_index,'cface_end_amp_tstat_ha'] = outcomes['end_amp_tstat']['ha']
                t.at[t_index,'cface_end_amp_tstat_an'] = outcomes['end_amp_tstat']['an']

                t.at[t_index,'cface_amp_diffXcond_ha'] = list(outcomes['amp_diffXcond']['ha'])
                t.at[t_index,'cface_amp_diffXcond_an'] = list(outcomes['amp_diffXcond']['an'])


                for emot in emots:
                    tspc0[emot][t_index,:,:] = outcomes['ts_pca'][emot][:,:,0]
                tsau12[t_index,:,:] = outcomes['ts']['ha'][:,:,ha_AU_index]

                for emot in emots:
                    for cong_or_incong in ['cong','incong']:
                        t.at[t_index,f'cface_cent_ts_{emot}_{cong_or_incong}_pca0'] = list(outcomes['aus_trial_pca_cond_cent'][emot][cong_or_incong])
                #acface_utils.plot_this_au_trial(outcomes['other_metrics']['an'],'an - comp0',times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps,plot_relevant_timestamps=False,results=outcomes['other_metrics']['an'])         
            else:
                t.at[t_index,'cface_goodwebcam']=False
    t['cface_outliers'] = ~(t.use_cface & t.cface_goodwebcam & (t.cface_latencies_validperc_ha>min_latency_validperc) & (t.cface_durations_cent_ha < max_response_duration)) 


    #OPTIONAL: Try to equalize age across both groups. Make new grouping column 'group06'. In each iteration, remove one subject from whichever group has more subjects. Remove the youngest subject from the clinical group, or remove the oldest subject from the control group. Results: Originally n=32/28, after getting p>0.05 it is 25/25 (saved group05_p05), after getting p>0.5 it is 22/21 (saved group05_p5)
    print("Equalizing age by removing subjects")
    valid_indices = t.use_cface & (t[group]!='') & ~t.cface_outliers & ~t.cface_bad_mri #default
    #valid_indices = t.use_cface & (t[group]!='') & ~t.cface_bad_mri
    t2 = t.loc[valid_indices,:]
    print("Equalizing age by removing subjects")
    t2['group06']=t2.loc[:,'group05']
    t2['temp'] = False
    for i in range(20):
        if sum(t2['group06']==gps[0]) > sum(t2['group06']==gps[1]):
            gp = 0
        else:
            gp = 1
        print(f"\nRemoving subject from {gps[gp]}")
        group_bool = t2['group06']==gps[gp]
        ages = t2.loc[group_bool,'age_years']
        if gp==0:
            target_age = ages.min()        
        else:
            target_age = ages.max()
        remove_this_subject = group_bool & (t2.age_years==target_age)
        index = remove_this_subject.index[remove_this_subject][0] #remove the first matching participant found
        t2.loc[index, 'group06'] = ''
        t2.loc[index,'temp']=True
        print(f"{gps[0]} n={sum(t2['group06']==gps[0])}, {gps[1]} n={sum(t2['group06']==gps[1])}")
        pval = acommonfuncs.plot_group_differences(['age_years'],t2,'group06',gps,print_means=True,print_tstat=False,plot=False)
        if pval>0.05 and ('exclude_p05' not in t2.columns):
            t2['exclude_p05'] = t2.loc[:,'temp']
        if pval>0.5 and ('exclude_p5' not in t2.columns):
            t2['exclude_p5'] = t2.loc[:,'temp']
            print('BREAK')
            break
    t['cface_exclude_p05'] = False
    t['cface_exclude_p5'] = False
    t.loc[valid_indices,'cface_exclude_p05'] = t2['exclude_p05']
    t.loc[valid_indices,'cface_exclude_p5'] = t2['exclude_p5']


    if save_table: 
        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_cface_face.csv')


### Start script here

print('Excluded participants due to no pulse')
print(t.loc[t.use_cface & ~t.cface_hasPulse,'subject'])
print('Excluded participants due to bad webcam data')
print(t.loc[t.use_cface & ~t.cface_goodwebcam,'subject'])
print(f'Excluded participants due to valid latencies calculable for < {min_latency_validperc}% of trials')
print(t.loc[t.use_cface & (t.cface_latencies_validperc_ha<min_latency_validperc),'subject'])
print(f'Excluded participants due to response duration > {max_response_duration} seconds')
print(t.loc[t.use_cface & (t.cface_durations_cent_ha > max_response_duration),'subject'])

"""
gps = [i for i in t[group].unique() if i!='']
if gps[0]=='hc': #make sure healthy controls are not first in the list
    gps = [gps[1],gps[0]]
print('gps',gps)
t.cface_bad_mri = t.record_id.isin(bad_mri_subjects)
"""

def print_group_counts(indices,title):
    #print number of subjects in each group, masked by 'indices'
    print(f"{indices.sum()}: {gps[0]} {(indices & (t[group]==gps[0])).sum()}, {gps[1]} {(indices & (t[group]==gps[1])).sum()}, na {(indices & (t[group]=='')).sum()} \t\t{title}")

print_group_counts(t.use_cface, 'use_cface (include & valid_cfacei & valid_cfaceo)')
print_group_counts(t[group]!='', {group})
print_group_counts(t.cface_bad_mri, 'bad_mri_subjects')
print_group_counts(t.use_cface & t.cface_outliers, 'cface_outliers')
print_group_counts(t.use_cface & (t[group]!=''), f'use_cface & {group}')
print_group_counts(t.use_cface & (t[group]!='') & ~t.cface_outliers, f'use_cface & {group} & not_outliers')
print_group_counts(t.use_cface & (t[group]!='') & ~t.cface_outliers & ~t.cface_bad_mri, f'use_cface & {group} & not_outliers & not_bad_mri')
print_group_counts(t.use_cface & (t[group]!='') & ~t.cface_outliers & ~t.cface_bad_mri & ~t.cface_exclude_p05, f'use_cface & {group} & not_outliers & not_bad_mri & age-matched p>0.05')

valid_indices = t.use_cface & (t[group]!='') & ~t.cface_outliers & ~t.cface_bad_mri #default
if age_matching: valid_indices = valid_indices & ~t.cface_exclude_p05

t2 = t.loc[valid_indices,:]

#Demographics
print('\nDemographics')
print(f"{gps[0]} n={sum(t2[group]==gps[0])}, {gps[1]} n={sum(t2[group]==gps[1])}")
acommonfuncs.plot_group_differences(['age_years','edu_num','fsiq2'],t2,group,gps,print_means=True,print_tstat=True,plot=False)


"""
#OPTIONAL: Try to equalize age across both groups. Make new grouping column 'group06'. In each iteration, remove one subject from whichever group has more subjects. Remove the youngest subject from the clinical group, or remove the oldest subject from the control group. Results: Originally n=32/28, after getting p>0.05 it is 25/25 (saved group05_p05), after getting p>0.5 it is 22/21 (saved group05_p5)
print("Equalizing age by removing subjects")
t2['group06']=t2.loc[:,'group05']
for i in range(20):
    if sum(t2['group06']==gps[0]) > sum(t2['group06']==gps[1]):
        gp = 0
    else:
        gp = 1
    print(f"\nRemoving subject from {gps[gp]}")
    group_bool = t2['group06']==gps[gp]
    ages = t2.loc[group_bool,'age_years']
    if gp==0:
        target_age = ages.min()        
    else:
        target_age = ages.max()
    remove_this_subject = group_bool & (t2.age_years==target_age)
    index = remove_this_subject.index[remove_this_subject][0] #remove the first matching participant found
    t2.loc[index, 'group06'] = ''
    print(f"{gps[0]} n={sum(t2['group06']==gps[0])}, {gps[1]} n={sum(t2['group06']==gps[1])}")
    pval = acommonfuncs.plot_group_differences(['age_years'],(2,2),t2,'group06',gps,print_means=True,print_tstat=False,plot=False)
    if pval>0.05 and ('group05_p05' not in t2.columns):
        t2['group05_p05'] = t2['group06']
    if pval>0.5 and ('group05_p5' not in t2.columns):
        t2['group05_p5'] = t2['group06']
        print('BREAK')
        break
t['cface_group05_p05'] = ''
t['cface_group05_p5'] = ''
t.loc[valid_indices,'cface_group05_p05'] = t2['group05_p05']
t.loc[valid_indices,'cface_group05_p5'] = t2['group05_p5']

#append the new columns to the outcomes_generic.csv file
filepath = f'{temp_folder}\\outcomes_cface_face.csv'
new_filepath = f'{temp_folder}\\outcomes_cface_face_new.csv'
new_columns = t.loc[:,['cface_group05_p05','cface_group05_p5']]
if os.path.exists(filepath):
    t3 = pd.read_csv(filepath)
    t3 = t3.merge(new_columns,how='left',left_index=True,right_index=True)
    t3.to_csv(new_filepath,index=False)
assert(0)
"""

table = acommonfuncs.get_male_table(t2,group,gps)
acommonfuncs.get_print_chi2(table,gps,'Male')
table = acommonfuncs.get_smoking_table(t2,group,gps)
acommonfuncs.get_print_chi2(table,gps,'Smoking')
for gp in gps:
    print(f"{gp} {'ethnicities'}")
    for item in np.unique(t2.loc[t2[group]==gp,'ethnicity']):
        print(f"\t{item}: n={np.sum((t2[group]==gp) & (t2['ethnicity']==item))}")     
variables_clinical_continuous = ['panss_P','panss_N','panss_G','panss_bluntedaffect','hamd','meds_chlor','cgi_s','sofas','sas']
for variable in variables_clinical_continuous:
    gp='cc'
    print(f"{variable} in {gp}: mean {t2.loc[t2[group]==gp,variable].mean():.3f}, std {t2.loc[t2[group]==gp,variable].std():.3f}")
print(f"Clinical group has {np.sum(t2.group01=='sz')} sz, {np.sum(t2.group01=='sza')} sza ")

#### Plots using the per-trial time series
if not load_table:
    tspc0 = {key:value[valid_indices,:,:] for key,value in tspc0.items()}
    tsau12 = tsau12[valid_indices,:,:]
    df = acommonfuncs.get_beh_data('cface1','134','out',use_MRI_task=False,source_data_raw=False) #Get behavioural data from any subject's *out.csv
    dfs={}
    dfs['ha'] = df.loc[df.ptemot=='HA',:]
    dfs['an'] = df.loc[df.ptemot=='AN',:]

    #Plot time series for first 16 subjects for trial ntrial, in a 4 x 4 grid using subplots. Also include relevant timestamps and labels on horizontal axis
    for emot in emots:
        x = tspc0[emot]
        ntrial = 30
        nrows = 5
        fig,axs=plt.subplots(nrows,nrows)
        for nsub in range(nrows**2):
            ax=axs[nsub//nrows,nsub%nrows]
            ts = x[nsub,ntrial,:]
            is_good_timeseries = acface_utils.good_timeseries(ts)
            ax.plot(times_trial_regular,ts,color='black',linewidth=0.4)
            acface_utils.plot_timestamps(ax)
            ax.set_title(f'sub {t2.subject.iloc[nsub]}, good?: {is_good_timeseries}')
        fig.suptitle(f'{emot} PCA component 0 time series for trial {ntrial}')
        plt.show(block=False)

    #Plot median (across trials) time series for first 16 subjects, in a 4 x 4 grid using subplots
    emot = 'ha'
    x=tspc0[emot] #numpy array of shape (n_subjects,n_trials,n_timepoints)
    xmtrials = np.median(x,axis=1) #central tendency across trials
    nrows = 5
    fig,axs=plt.subplots(nrows,nrows)
    for nsub in range(nrows**2):
        ax=axs[nsub//nrows,nsub%nrows]
        ax.plot(times_trial_regular,xmtrials[nsub,:],color='black',linewidth=0.4)
        acface_utils.plot_timestamps(ax)
        ax.set_title(f'sub {t2.subject.iloc[nsub]}')
    fig.suptitle(f'{emot} PCA component 0 time series: median across trials')
    plt.show(block=False)

    #Plot median congruent and median incongruent response for each subject (overlaid). Overlay the median (across subjects). Separately for each instruction and group
    colors = ['blue','red']
    fig,axs=plt.subplots(2,2)
    for i,emot in enumerate(emots):
        x=tspc0[emot]
        for j,gp in enumerate(gps):
            ax=axs[i,j]
            x2 = x[t2[group]==gp,:,:] #data for one group. Array of shape (n_subjects,n_trials,n_timepoints)
            for cong,color in enumerate(colors):
                x2m = np.median(x2[:,dfs[emot].cong == cong,:], axis=1)
                for nsub in range(x2m.shape[0]):
                    ax.plot(times_trial_regular,x2m[nsub,:],color=color,linewidth=0.2) 
                ax.plot(times_trial_regular,np.median(x2m,axis=0),color=color,linewidth=1.6) #median across subjects
            ax.set_title(f'{gp} {emot}')
            ax.set_ylim([-3,3])
            acface_utils.plot_timestamps(ax)
    fig.suptitle(f'PCA component 0 time series: median (cong red or incong blue) response for each subject')
    plt.show(block=False)

    #Plot median (across subjects) of (median congruent - median incongruent response). Separately for each instruction and group
    fig,axs=plt.subplots(2,2)
    for i,emot in enumerate(emots):
        x=tspc0[emot]
        for j,gp in enumerate(gps):
            x2 = x[t2[group]==gp,:,:] #data for one group. Array of shape (n_subjects,n_trials,n_timepoints)
            x2d = np.median(x2[:,dfs[emot].cong == 1,:], axis=1) - np.median(x2[:,dfs[emot].cong == 0,:], axis=1) #median congruent - median incongruent response, for each subject, for each timepoint
            x2dmsubs = np.median(x2d,axis=0) #median across subjects of above differences
            ax=axs[i,j]
            ax.plot(times_trial_regular,x2dmsubs,color='black',linewidth=0.4)
            ax.set_title(f'{gp} {emot}')
            acface_utils.plot_timestamps(ax)
    fig.suptitle(f'PCA component 0 time series: median across subjects of \n(median congruent - median incongruent response)')
    plt.show(block=False)

#### Plots using measures averaged across trials

variables = ['cface_amp_max_cent','cface_amp_max_slope'] + ['cface_latencies_validperc_ha','cface_latencies_cent_ha','cface_durations_cent_ha','cface_maxgrads_cent_ha','cface_maxgradns_cent_ha','cface_end_amp_tstat_ha'] + ['cface_latencies_validperc_an','cface_latencies_cent_an','cface_durations_cent_an','cface_maxgrads_cent_an','cface_maxgradns_cent_an','cface_end_amp_tstat_an'] + ['cface_maxgrads_cent_ha_AU12']
print("\nGroup differences in face expression measures")
acommonfuncs.plot_group_differences(['cface_amp_max_cent','cface_amp_max_slope'],t2,group,gps,print_means=True,print_tstat=True)
acommonfuncs.plot_group_differences(['cface_latencies_validperc_ha','cface_latencies_cent_ha','cface_durations_cent_ha','cface_maxgrads_cent_ha','cface_maxgradns_cent_ha','cface_maxgrads_cent_ha_AU12','cface_end_amp_tstat_ha'],t2,group,gps,print_means=True,print_tstat=True)
acommonfuncs.plot_group_differences(['cface_latencies_validperc_an','cface_latencies_cent_an','cface_durations_cent_an','cface_maxgrads_cent_an','cface_maxgradns_cent_an','cface_end_amp_tstat_an'],t2,group,gps,print_means=True,print_tstat=True)

#Plot smile amplitude group difference

t2.loc[t2[group]=='cc',group] = 'clinical'
t2.loc[t2[group]=='hc',group] = 'healthy'
fig,ax=plt.subplots(figsize=(3,2.5))
sns.stripplot(ax=ax, data = t2, x=group, hue=group,y='cface_amp_max_cent',zorder=1,alpha=0.5,palette=colors)
box=sns.boxplot(ax=ax, data=t2,y='cface_amp_max_cent',x=group, orient = 'v',fliersize=0,width=0.4,zorder=2,
boxprops={'facecolor':'none'})
ax.set_ylabel('Smile amplitude')
ax.set_xlabel('')
fig.tight_layout()
plt.show(block=False)
assert(0)


t2_patients =  t2.loc[t2[group]==gps[0],:]
print(f'\nCorrelations within {gps[0]} only:')
acommonfuncs.pairwise_correlations2(['fsiq2','panss_P','panss_N','panss_G','panss_bluntedaffect','cgi_s','sofas','meds_chlor'], variables,t2_patients)
#acommonfuncs.pairwise_correlations2(['cgi_s'],['cface_amp_max_cent','cface_latencies_cent_ha','cface_latencies_cent_an'],t2_patients,to_plot=True) #plot the significant ones

#one-sample t-test of end_amp_tstat values
for emot in emots:
    for gp in gps:
        vals = t2.loc[t2[group]==gp,f'cface_end_amp_tstat_{emot}'].values
        stat = pg.ttest(vals,0,alternative='two-sided')
        print(f"end_amp_tstat {emot} {gp} 1 sample ttest t={stat['T'].iloc[0]:.3f}, p={stat['p-val'].iloc[0]:.3f}")

#Look at cent time series
def plot_cent_ts(ax,dataframe,title,ylims=None):
    for t_index in range(len(dataframe)):
        data = dataframe.iloc[t_index]
        ax.plot(times_trial_regular,data,color='black',linewidth=0.4)
    for j in relevant_timestamps:
        ax.axvline(x=j) 
    for i,annotation in enumerate(relevant_labels):
        ax.text(midpoint_timestamps[i],-0.5,annotation[0:4],ha='center')
    ax.set_title(title)
    ax.set_ylim(ylims)

#Plot cent time series for each instruction
fig,axs=plt.subplots(2,3)
for i,gp in enumerate(gps):
    gp_bool = t2[group]==gp
    plot_cent_ts(axs[i,0],t2.loc[gp_bool,'cface_cent_ts_ha_au12'],f'{gp} - ha - AU12',ylims=[0,4])
    plot_cent_ts(axs[i,1],t2.loc[gp_bool,'cface_cent_ts_ha_pca0'],f'{gp} - ha - pca0',ylims=[-2.5,2.5])
    plot_cent_ts(axs[i,2],t2.loc[gp_bool,'cface_cent_ts_an_pca0'],f'{gp} - an - pca0',ylims=[-2.5,2.5])
fig.suptitle('cface_cent_ts')
fig.tight_layout()
plt.show(block=False)

#Plot cent time series for each instruction, separately for congruent and incongruent
gp='hc'
gp_bool = t2[group]==gp
fig,axs=plt.subplots(2,2)
for i,emot in enumerate(emots):
    for j,cong_or_incong in enumerate(['cong','incong']):
        plot_cent_ts(axs[i,j],t2.loc[gp_bool,f'cface_cent_ts_{emot}_{cong_or_incong}_pca0'],f'{gp} {emot} {cong_or_incong}',ylims=[-2.5,2.5])
fig.suptitle('cface_cent_ts_pca0: congruent or incongruent')
fig.tight_layout()
plt.show(block=False)

def onesamp_ttest_columns(array):
    """
    For each column of an array, do a one-sample t-test against zero, and return the T-statistic
    """
    return np.array([pg.ttest(array[:,i],0,alternative='two-sided')['T'].iloc[0] for i in range(array.shape[1])])

#For each participant, find difference between their cent congruent and cent incongruent time series. Get one such difference time series for each participant. Do t-test for deviation from zero (cong>incong), separately at each timepoint.
fig,axs = plt.subplots(2,2)
for i,emot in enumerate(emots):
    for j,gp in enumerate(gps):
        ts_cong=np.stack(t2.loc[t2[group]==gp,f'cface_cent_ts_{emot}_cong_pca0'].values)
        ts_incong=np.stack(t2.loc[t2[group]==gp,f'cface_cent_ts_{emot}_incong_pca0'].values)
        ts_diff = ts_cong - ts_incong
        vals = onesamp_ttest_columns(ts_diff)
        vals_df = pd.DataFrame([[list(vals)]]).iloc[:,0]
        plot_cent_ts(axs[i,j],vals_df,f'{gp} {emot}',ylims=[-2.5,2.5])

        from scipy import stats
        t_criticalvalue = stats.t.ppf(q=1-0.05/2,df=ts_diff.shape[0]-1)
        axs[i,j].axhline(y=t_criticalvalue,color='red')
        axs[i,j].axhline(y=-t_criticalvalue,color='red')
fig.suptitle('cong-incong tstat')
fig.tight_layout()
plt.show(block=False)

if not load_table:
    #For each participant, for each emotion, for each time point in the trial, we have obtained a t-statistic for  (congruent-incongruent) smile amplitude (where amplitudes are normalized by the value at the start of the stimulus movement). Plot the mean (across participants) t-statistic, for each emotion, for each group
    fig,axs=plt.subplots(2,2)
    for i,emot in enumerate(emots):
        for j,gp in enumerate(gps):
            ts_emot=np.stack(t2.loc[t2[group]==gp,f'cface_amp_diffXcond_{emot}'].values)
            vals = onesamp_ttest_columns(ts_emot)
            vals_df = pd.DataFrame([[list(vals)]]).iloc[:,0]
            plot_cent_ts(axs[i,j],vals_df,f'{gp} {emot}',ylims=[-2.5,2.5])
    fig.suptitle('cong-incong tstat')
    fig.tight_layout()
    plt.show(block=False)