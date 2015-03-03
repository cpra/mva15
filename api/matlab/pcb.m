
classdef pcb < handle
    % A printed circuit board.
    % ~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology

    properties
        % Root directory path.
        root = '';
        
        % Scale factor.
        scale = 1;
        
        % Struct of id : image_path pairs.
        recordings_ = struct();
        
        % Struct of id : crop pairs (cropinfo cache).
        cache_cropinfo = struct();
    end
    
    methods
        function obj = pcb(root, scale)
            % Constructor.
            % root: root directory path.
            % scale: scale factor (1 = original size).
            
            if exist(root, 'dir') ~= 7
                error('%s is not a directory', root);
            end
            
            if scale <= 0 || scale > 2
                error('Scale must be > 0 and <= 2');
            end
            
            obj.root = root;
            obj.scale = scale;
            
            d = dir(root);
            isub = not([d(:).isdir]);
            nameFolds = {d(isub).name}';
            
            isrec = cellfun(@(x) strcmp(x(1:3), 'rec'), nameFolds);
            isjpg = cellfun(@(x) strcmp(x(end-2:end), 'jpg'), nameFolds);
            
            nameFolds = nameFolds(and(isrec, isjpg));
            
            for i = 1:numel(nameFolds)
                n = nameFolds{i};
                obj.recordings_.(n(1:end-4)) = fullfile(root, n);
            end
        end
        
        function ret = id(obj)
            % Returns the PCB ID.
            
            [~, n, ~] = fileparts(obj.root);
            ret = str2double(n(4:end));
        end
        
        function ids = recordings(obj)
            % Returns a list of IDs of all available recordings.
            
            ids = fieldnames(obj.recordings_);
            ids = cellfun(@(x) str2double(x(4:end)), ids);
        end
        
        function im = image(obj, rec)
            % Returns the image of the specified recording.
            % rec: desired recording (see recordings()).
            
            key = sprintf('rec%d', rec);
            if isfield(obj.recordings_, key)
                im = imread(obj.recordings_.(key));
                if obj.scale ~= 1
                    im = imresize(im, obj.scale);
                end
            else
                error('Recording %d does not exist', rec);
            end
        end
        
        function im = mask(obj, rec)
            % Returns the mask of the specified recording.
            % rec: desired recording (see recordings()).
            
            key = sprintf('rec%d', rec);
            if isfield(obj.recordings_, key)
                [p, ~, ~] = fileparts(obj.recordings_.(key));
                fn = fullfile(p, sprintf('%s-mask.png', key));
                
                im = imread(fn);
                if obj.scale ~= 1
                    im = imresize(im, obj.scale);
                end
                
                im = im > 0;
            else
                error('Recording %d does not exist', rec);
            end
        end
        
        function im = image_masked(obj, rec)
            % Returns the image of the specified recording, masked by the corresponding mask and cropped to remove background.
            % rec: desired recording (see recordings()).
            
            im = obj.image(rec);
            mask = obj.mask(rec);
            
            mask = repmat(mask, [1 1 size(im, 3)]);
            im(not(mask)) = 0;
            
            ci = obj.cropinfo(rec);
            im = im(ci(1):ci(1)+ci(3), ci(2):ci(2)+ci(4), :);
        end
        
        function ret = ics(obj, rec, cropped, size, aspect)
            % Returns a list of IC chips as a list of Annot objects.
            % rec: desired recording (see recordings()).
            % cropped: whether to return coordinates for cropped images (see image_masked()).
            % size: (min, max) size of returned ICs in cm^2, disregarding the scale factor (0 = all).
            % aspect: (min, max) aspect ratio of returned ICs (0 = all).
            
            key = sprintf('rec%d', rec);
            if isfield(obj.recordings_, key)
                [p, ~, ~] = fileparts(obj.recordings_.(key));
                fn = fullfile(p, sprintf('%s-annot.txt', key));
                if exist(fn, 'file') ~= 2
                    error('File %s does not exist', fn);
                end
                
                ret = {};
                fid = fopen(fn);
                tline = fgetl(fid);
                while ischar(tline)
                    % parse line
                    
                    t = textscan(tline, '%f %f %f %f %f %[^\n]');
                    
                    text = '';
                    if ~isempty(t{6})
                        text = t{6};
                    end
                    
                    rect = [t{1} t{2} t{3} t{4} t{5}];
                    
                    % check dimensions
                    
                    sz = [rect(3)/87.4, rect(4)/87.4];
                    asp = max(sz) / min(sz);
                    sz = sz(1) * sz(2);
                    
                    ok = true;
                    
                    if size(1) > 0 && sz < size(1)
                        ok = false;
                    end
                    
                    if size(2) > 0 && sz > size(2)
                        ok = false;
                    end
                    
                    if aspect(1) > 0 && asp < aspect(1)
                        ok = false;
                    end
                    
                    if aspect(2) > 0 && asp > aspect(2)
                        ok = false;
                    end
                    
                    if obj.scale ~= 1
                        rect(1:4) = rect(1:4) * obj.scale;
                    end
                    
                    if cropped
                        ci = obj.cropinfo(rec);
                        rect(1) = rect(1) - ci(2) + 1;
                        rect(2) = rect(2) - ci(1) + 1;
                    end
                    
                    % add to list
                    
                    if ok
                        ret{end+1} = annot(rect, obj.scale, text);    
                    end
                    
                    tline = fgetl(fid);
                end
                fclose(fid);
            else
                error('Recording %d does not exist', rec);
            end
        end
        
        function ci = cropinfo(obj, rec)
            % Return (and cache) information for auto cropping a PCB image.
            % rec: desired recording (see recordings()).
            
            key = sprintf('rec%d', rec);
            if isfield(obj.cache_cropinfo, key)
                ci = obj.cache_cropinfo.(key);
                return;
            end
            
            im = obj.mask(rec);
            rp = regionprops(im, 'Area', 'BoundingBox');
            
            smax = find([rp.Area] == max([rp.Area]), 1);
            rp = rp(smax);
            
            bb = rp.BoundingBox;
            obj.cache_cropinfo.(key) = round([bb(2) bb(1) bb(4) bb(3)]);
            
            ci = obj.cache_cropinfo.(key);
        end
    end
    
end

