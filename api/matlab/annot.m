
classdef annot < handle  
    % An annotated PCB component.
    % ~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology
    
    properties
        % region of component as [cx, cy, dx, dy, angle].
        rect = [0 0 0 0 0];
        
        % scale factor.
        scale = 1;
        
        % optional label text.
        text = '';
    end
    
    methods
        function obj = annot(rect, scale, text)
            % Constructor.
            % rect: egion of component as [cx, cy, dx, dy, angle].
            % scale: scale factor.
            % text: optional label text.
            
            obj.rect = rect;
            obj.scale = scale;
            obj.text = text;
        end
        
        function sz = size_pixels(obj, scaled)
            % Returns the size of the component in pixels.
            % scaled: whether to regard the scale factor.
            
            sz = obj.rect(3)*obj.rect(4);
            if ~scaled
                sz = sz / obj.scale;
            end
        end
        
        function sz = size_cm2(obj, scaled)
            % Returns the size of the component in cm^2.
            % scaled: whether to regard the scale factor.
            
            sz = (obj.rect(3)/87.4) * (obj.rect(4)/87.4);
            if ~scaled
                sz = sz / obj.scale;
            end
        end
        
        function as = aspect(obj)
            % Returns the aspect ratio (larger side length / smaller side length).
            
            as = max([obj.rect(3), obj.rect(4)]) / min([obj.rect(3), obj.rect(4)]);
        end
    end
end
