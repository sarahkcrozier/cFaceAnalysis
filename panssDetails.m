function options = pcnsDetails_change_name(options)

design jasp spreadsheet 
extract data
chack correct periods to extract 

prepare data table

options.quest.name = "PANSS";
options.quest.positiveSymptoms.items = ["panss_p1","panss_p2","panss_p3","panss_p4","panss_p5","panss_p6","panss_p7"];
options.quest(1).negativeSymptoms.items = ["panss_n1","panss_n2","panss_n3","panss_n4","panss_n5","panss_n6","panss_n7"];
options.quest(1).generalSymptoms.items = ["panss_g1","panss_g2","panss_g3","panss_g4","panss_g5","panss_g6","panss_g7","panss_g8","panss_g9","panss_g10","panss_g11","panss_g12","panss_g13","panss_g14","panss_g15","panss_g16"];
options.quest(1).bluntedAffect.item = ["panss_n1"];
options.quest(1).emotionalWithdrawal.items = ["panss_n2"];
for pilotorreal = 2, extract group .





end