classdef pcbdataset < handle
    % A PCB dataset.
    % ~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology

    properties
        % Root path to dataset.
        root = '';
        
        % Struct of pcbid : root_path pairs.
        pcbpaths = struct();
    end
    
    methods
        function obj = pcbdataset(root)
            % Constructor.
            % root: root path to the dataset.
            
            if exist(root, 'dir') ~= 7
                error('%s is not a directory', root);
            end
            
            obj.root = root;
            
            d = dir(root);
            isub = [d(:).isdir];
            nameFolds = {d(isub).name}';
            nameFolds(ismember(nameFolds,{'.','..'})) = [];
            
            for i = 1:numel(nameFolds)
                n = nameFolds{i};
                obj.pcbpaths.(n) = fullfile(root, n);
            end
        end
        
        function num = num_pcbs(obj)
            % Returns the number of PCBs in the dataset.
            
            num = numel(fieldnames(obj.pcbpaths));
        end
        
        function ids = pcb_ids(obj)
            % Returns a list of IDs of all PCBs in the dataset as a cell array.
            
            ids = fieldnames(obj.pcbpaths);
            ids = cellfun(@(x) str2double(x(4:end)), ids);
        end
        
        function p = get_pcb(obj, id, scale)
            % Returns the PCB with the given ID as a PCB object.
            % id: PCB id (see pcb_ids()).
            % scale: scale factor (1 = original size).
            
            key = sprintf('pcb%d', id);
            if isfield(obj.pcbpaths, key)
                p = pcb(obj.pcbpaths.(key), scale);
            else
                error('Unknown PCB ID');
            end
        end
    end
end
