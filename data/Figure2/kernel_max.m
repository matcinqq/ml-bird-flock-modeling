function [max_x, pd] = kernel_max(data,plt,bw)

% function [max_x, pd] = kernel_max(data,plt,bw)
%
% Uses MATLAB's fitdist function to find the modal value of a noisy
% distribution. 
% 
% data = input data as 1 column
% plt = true/false to create a histogram plot with fitted distribution
% bw = bandwidth of kernel, leave undefined to auto-select.  0.15
% reasonable for most bird flock data
%
% max_x = fitted modal value of data
% pd = fitdist object

if nargin > 2
    pd = fitdist(data,'kernel','Bandwidth',bw);
else
    pd = fitdist(data(:,1),'kernel');
end
x = min(data):0.001:max(data);
%x = min(data):1:max(data);
y = pdf(pd,x);
[max_y,idx] = max(y);
max_x= x(idx);

if plt
    figure; 
    %h = histogram(data,-2:0.2:2);
    h = histogram(data,20);
    hold on;
    plot(x,y./max(y).*max(h.Values));
end