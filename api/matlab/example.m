
% Demonstrates the use of the Matlab API.
% ~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology

close all; clear variables; clc;

% adapt the following variables
dataset = pcbdataset('/home/chris/projects/mva-paper/dataset/');
pcbid = 1;
recid = 1;

fprintf('Dataset contains images of %d PCBs\n', dataset.num_pcbs());
fprintf(' %s\n', mat2str(dataset.pcb_ids()));

pcb = dataset.get_pcb(pcbid, 0.3);
fprintf('Loaded PCB %d, available recordings: %s\n', pcb.id(), mat2str(pcb.recordings()));

ics = pcb.ics(recid, true, [0, 0], [0, 0]);
fprintf('PCB contains %d ICs\n', numel(ics));

im = pcb.image_masked(recid);

imshow(im);
for i = 1:numel(ics)
    ic = ics{i};
    
    p1 = [ic.rect(1)-ic.rect(3)/2, ic.rect(2)-ic.rect(4)/2];
    p2 = [ic.rect(1)+ic.rect(3)/2, ic.rect(2)-ic.rect(4)/2];
    p3 = [ic.rect(1)+ic.rect(3)/2, ic.rect(2)+ic.rect(4)/2];
    p4 = [ic.rect(1)-ic.rect(3)/2, ic.rect(2)+ic.rect(4)/2];
    
    ang = deg2rad(ic.rect(5));
    rmat = [cos(ang) -sin(ang) ; sin(ang) cos(ang)];
    
    pt = [p1 ; p2 ; p3 ; p4]';
    
    pt(1,:) = pt(1,:) - ic.rect(1);
    pt(2,:) = pt(2,:) - ic.rect(2);
    pt = rmat * pt;
    pt(1,:) = pt(1,:) + ic.rect(1);
    pt(2,:) = pt(2,:) + ic.rect(2);
    
    pt = [pt pt(:,1)];
    line(pt(1,:), pt(2,:), 'Color', 'g');
end
