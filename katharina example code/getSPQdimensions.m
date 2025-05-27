function getSPQdimensions

%% DEFINE SUBSCALES
options.paths.data = '/Volumes/Samsung_T5/SNG/projects/SAPS/data/';
options.paths.dbExport = [options.paths.data,'REDCapExport',filesep];
options.paths.questData = [options.paths.data,'questionnaires',filesep];
options.study.acronym = 'SAPS';
file = dir([options.paths.questData,options.study.acronym,'*.csv']);

data = readtable([options.paths.dbExport,file.name]);

for n = 1:max(data.record_id)
    psid = [options.study.acronym,'_',data.record_id];
id_rows   = find(data.record_id==n);
psDir = [options.paths.questData,psid,filesep];
mkdir(psDir);
screen_event  = intersect(id_rows,find(strcmp(data.redcap_event_name,'demograph')));
onlineQ_event = intersect(id_rows,find(strcmp(data.redcap_event_name,'details')));
expDay_event  = intersect(id_rows,find(strcmp(data.redcap_event_name,'clinical')));
spq_rows = intersect(id_rows,find(strcmp(data.redcap_repeat_instrument,'panss')));

%{
filter out id rows with cface i true/1 (this is an intersect)
spq etc is the column name in redcap
%}

IdeasOfReference       = data{spq_rows,["spq01","spq10","spq19","spq28","spq37","spq45","spq53","spq60","spq63"]};
ExcessiveSocialAnxiety = data{spq_rows,["spq02","spq11","spq20","spq29","spq38","spq46","spq54","spq71"]};
MagicalThinking        = data{spq_rows,["spq03","spq12","spq21","spq30","spq39","spq47","spq55"]};
UnusualPerceptualExperiences = data{spq_rows,["spq04","spq13","spq22","spq31","spq40","spq48","spq56","spq61","spq64"]};
EccentricBehaviour     = data{spq_rows,["spq05","spq14","spq23","spq32","spq67","spq70","spq74"]};
NoCloseFriends         = data{spq_rows,["spq06","spq15","spq24","spq33","spq41","spq49","spq57","spq62","spq66"]};
OddSpeech              = data{spq_rows,["spq07","spq16","spq25","spq34","spq42","spq50","spq58","spq69","spq72"]};
ConstrictedAffect      = data{spq_rows,["spq08","spq17","spq26","spq35","spq43","spq51","spq68","spq73"]};
Suspiciousness         = data{spq_rows,["spq09","spq18","spq27","spq36","spq44","spq52","spq59","spq65"]};

SPQmeansTable = table(mean(IdeasOfReference),mean(ExcessiveSocialAnxiety),mean(MagicalThinking),mean(UnusualPerceptualExperiences),...
    mean(EccentricBehaviour),mean(NoCloseFriends),mean(OddSpeech),mean(ConstrictedAffect),mean(Suspiciousness),'VariableNames',...
    {'IdeasOfReference','ExcessiveSocialAnxiety','MagicalThinking','UnusualPerceptualExperiences','EccentricBehaviour',...
    'NoCloseFriends','OddSpeech','ConstrictedAffect','Suspiciousness'});

% SAVE
save([psDir,'SPQmeansTable.mat'],SPQmeansTable); writematrix(SPQmeansTable,[psDir,'SPQmeansTable.csv']);


end

end