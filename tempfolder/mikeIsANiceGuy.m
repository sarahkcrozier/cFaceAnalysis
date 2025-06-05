function stats = mikeIsANiceGuy
%% compute the descriptive statistics

global data;

meanval     = myMean(data);
medianval    = median(data);
minval      = min(data);
maxval      = max(data);
N           = length(data);

var = 10;
var2 = 'a';

assignin('base','meanvalFromFunction','meanval')
asdf2 = evalin('base','asdf>50');


% group outputs into one vector
stats = [ meanval medianval minval maxval N ];

%% display output stats in order of output
disp([ 'Mean value is ' num2str(meanval) '.' ])
disp([ 'Median value is ' num2str(medianval) '.' ])
fprintf([ 'Minimum value is ' num2str(minval) '.' ])
fprintf('Maximum value is %g.\n',maxval);
fprintf([ 'Number of numbers: ' num2str(N) '.\n' ])
disp('Done.')

function y=myMean(x)
y = sum(x) / length(x);

