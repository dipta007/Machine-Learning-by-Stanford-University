clc;
close all;
clear all;
for(m=1:55)
    n=m-16;
    x(m)=n;
    if(x(m)=0)
        y(m)=1;
    else
        y(m)=0;
    end;
end;
stem(x(m);