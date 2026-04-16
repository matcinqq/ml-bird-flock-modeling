clear
load('Figure2Data.mat')
% Neighbor position data are stored in the variable nxyz1, where each row is a set of 
% nearest-neighbor positions with X (first column) oriented perpendicular to the focal bird's direction 
% of travel, Y (second column) oriented along the direction of travel, and Z (third column) oriented along 
% the axis of gravity.

% PLOT 1-D HISTOGRAM WITHIN SLICE
figure; set(gcf,'position',[-1300,300,956,671]);
set(gcf,'color','w');

% Get Data within +/- 1 wingspan
idx = find( abs(nxyz1(:,3)) < 1 & rnorm(nxyz1) > (0.8) & nxyz1(:,2)>0); %Find data within a slice;
idx = idx(1:5:end); %Take one data point for each wingbeat
subplot(3,4,9:10);
h = histogram(nxyz1(idx,1),-2.4:0.20:2.4);
set(h,'FaceColor',[0.7,0.7,0.7]);
hold on;

[max_x, pd] = kernel_max(nxyz1(idx,1),0,0.2)
x = -2.5:0.001:2.5;
y = pdf(pd,x);
[max_y,idx2] = max(y);
max_x= x(idx2);
plot(x,y./max(y).*max(h.Values));

ylabel('Count');
xlim([0,2.0]);
ylim([300, 700])
set(gca,'TickDir','out');
a = gca;
set(a,'box','off','color','none')
b = axes('Position',get(gca,'Position'),'box','on','xtick',[],'ytick',[]);
axes(a);
xlabel('Lateral distance (WS)');

%PLOT 2D SCATTER PLOT In SLICE
subplot(3,4,[1,2,5,6]);
idx = find( abs(nxyz1(:,3)) < WS*1 & rnorm(nxyz1) > (WS*0.8) & nxyz1(:,2)>0);
[h,c] = hist3([nxyz1(idx,1),nxyz1(idx,2)],{-5.1:0.2:5.1,-0.1:0.2:10.1});
[x,y] = meshgrid(c{1},c{2});
surf(x,y,h')
set(get(gca,'child'),'FaceColor','interp','EdgeColor','interp','CDataMode','auto'); view(2)
xlim([0, 2.0]); ylim([0,2.0]);

ylabel('Forward distance (WS)');

set(gca,'TickDir','out');
set(gca,'xticklabel',[])
a = gca;
set(a,'box','off','color','none')
b = axes('Position',get(gca,'Position'),'box','on','xtick',[],'ytick',[]);
axes(a);
set(gca, 'ydir','reverse')

% PLOT 1-D HISTOGRAM OUT OF SLICE
idx = find(abs(nxyz1(:,3))>1  & nxyz1(:,2)>0);
idx = idx(1:10:end);
subplot(3,4,11:12); cla
h = histogram(abs(nxyz1(idx,1)),-2.4:0.20:2.4);
set(h,'FaceColor',[0.7,0.7,0.7]);
hold on;



[max_x, pd] = kernel_max(nxyz1(idx,1),0,0.30)
x = -2.5:0.001:2.5;
y = pdf(pd,x);
[max_y,idx2] = max(y);
max_x= x(idx2);
plot(x,y./max(y).*max(h.Values));
set(h,'FaceColor',[0.7,0.7,0.7]);
%xlabel('Right-left distance (m)');

ylabel('Count');
xlim([0,2.0]);
ylim([0,1000]);
set(gca,'TickDir','out');
a = gca;
set(a,'box','off','color','none')
b = axes('Position',get(gca,'Position'),'box','on','xtick',[],'ytick',[]);
axes(a);
xlabel('Lateral distance (WS)');

%PLOT 2D SCATTER PLOT OUT OF SLICE
subplot(3,4,[3,4,7,8]); cla
[h,c] = hist3([abs(nxyz1(idx,1)),nxyz1(idx,2)],{-5.1:0.2:5.1,-0.1:0.2:10.1});
[x,y] = meshgrid(c{1},c{2});
surf(x,y,h')
set(get(gca,'child'),'FaceColor','interp','EdgeColor','interp','CDataMode','auto'); view(2)
xlim([0, 2.0]); ylim([0,2.0]);

ylabel('Forward distance (WS)');

set(gca,'TickDir','out');
set(gca,'xticklabel',[])
a = gca;
set(a,'box','off','color','none')
b = axes('Position',get(gca,'Position'),'box','on','xtick',[],'ytick',[]);
axes(a);
set(gca, 'ydir','reverse')

