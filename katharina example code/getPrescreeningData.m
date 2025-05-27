function getPrescreeningData

options = specifyOptions;
options = getQuestionnaireDetails(options);

%% LOAD REDCap datafile
file = dir([options.paths.DBExport,'*.csv']);
data = readtable([options.paths.DBExport,file(2).name]);

%% EXTRACT data

% loop through REDCap IDs
for n = 1:max(data.record_id)
    id_rows   = find(data.record_id==n);
    spq_rows  = intersect(id_rows,find(strcmp(data.redcap_repeat_instrument,'spq')));
    if numel(spq_rows)>1
        spq1_rows = intersect(spq_rows,find(data.redcap_repeat_instance==1));
    else
        spq1_rows = spq_rows;
    end

    if ~isempty(spq_rows)
        IdeasOfReference(n,:)             = data{spq1_rows,options.quest(1).ideasOfReference.items};
        ExcessiveSocialAnxiety(n,:)       = data{spq1_rows,options.quest(1).excessiveSocialAnxiety.items};
        MagicalThinking(n,:)              = data{spq1_rows,options.quest(1).magicalThinking.items};
        UnusualPerceptualExperiences(n,:) = data{spq1_rows,options.quest(1).unusualPerceptualExperiences.items};
        EccentricBehaviour(n,:) = data{spq1_rows,options.quest(1).eccentricBehaviour.items};
        NoCloseFriends(n,:)     = data{spq1_rows,options.quest(1).noCloseFriends.items};
        OddSpeech(n,:)          = data{spq1_rows,options.quest(1).oddSpeech.items};
        ConstrictedAffect(n,:)  = data{spq1_rows,options.quest(1).constrictedAffect.items};
        Suspiciousness(n,:)     = data{spq1_rows,options.quest(1).suspiciousness.items};
    else
        IdeasOfReference(n,:)             = NaN(1,size(options.quest(1).ideasOfReference.items,2));
        ExcessiveSocialAnxiety(n,:)       = NaN(1,size(options.quest(1).excessiveSocialAnxiety.items,2));
        MagicalThinking(n,:)              = NaN(1,size(options.quest(1).magicalThinking.items,2));
        UnusualPerceptualExperiences(n,:) = NaN(1,size(options.quest(1).unusualPerceptualExperiences.items,2));
        EccentricBehaviour(n,:) = NaN(1,size(options.quest(1).eccentricBehaviour.items,2));
        NoCloseFriends(n,:)     = NaN(1,size(options.quest(1).noCloseFriends.items,2));
        OddSpeech(n,:)          = NaN(1,size(options.quest(1).oddSpeech.items,2));
        ConstrictedAffect(n,:)  = NaN(1,size(options.quest(1).constrictedAffect.items,2));
        Suspiciousness(n,:)     = NaN(1,size(options.quest(1).suspiciousness.items,2));
    end
end

id = [1:max(data.record_id)]';

SPQmeansTable = table(id,mean(IdeasOfReference,2),mean(ExcessiveSocialAnxiety,2),mean(MagicalThinking,2),mean(UnusualPerceptualExperiences,2),...
    mean(EccentricBehaviour,2),mean(NoCloseFriends,2),mean(OddSpeech,2),mean(ConstrictedAffect,2),mean(Suspiciousness,2),'VariableNames',...
    {'ID','IdeasOfReference','ExcessiveSocialAnxiety','MagicalThinking','UnusualPerceptualExperiences','EccentricBehaviour',...
    'NoCloseFriends','OddSpeech','ConstrictedAffect','Suspiciousness'});

% SAVE
save([options.paths.questData,'SPQmeansTable.mat'],'SPQmeansTable'); writetable(SPQmeansTable,[options.paths.questData,'SPQmeansTable.csv']);


gender  = NaN(max(data.record_id),1);
age     = NaN(max(data.record_id),1);
edu_yrs = NaN(max(data.record_id),1);
constrAffect = NaN(max(data.record_id),1);
email   = cell(max(data.record_id),1);

% get prescreening data table
for n = 1:max(data.record_id)
    id_rows   = find(data.record_id==n);
    for i = 1:numel(id_rows)
        idRow = id_rows(i);
        if ~isnan(data.gender(idRow))
            if ~isempty(data.email{idRow})
                email{n,:} = data.email{idRow};
            end
            gender(n,:) = data.gender(idRow);
            age(n,:) = data.age(idRow);
            edu_yrs(n,:) = data.edu_yrs(idRow);
            constrAffect(n,:) = SPQmeansTable.ConstrictedAffect(n);
        end
    end
end
PrescreeningTable = table(id,email, gender, age, edu_yrs,constrAffect,'VariableNames',...
    {'ID','email','gender','age','education in yrs','constricted affect'});
writetable(PrescreeningTable,[options.paths.questData,'PrescreeningTable.csv'])
end