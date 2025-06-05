random = rand(1)

isGreater = random > 0.5

if isGreater == 0
    disp([ 'The value is ' num2str(random) 'and is not bigger than .5.' ])

else
    disp([ 'The value is ' num2str(random) 'and is  bigger than .5.'] )
end
