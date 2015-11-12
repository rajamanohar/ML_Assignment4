classdef SMOAlgo
    properties
        DsS=[];      % S = {(xi, yi); yi = ±1;i = 1, . . . , l}
        Kij=[];      % Kij = K(xi,xj)=xi·xj
        Alpha=[];    %  ? = {?1, . . . , ?l} randomly subject to constraint  yi?i = 0
        yi=[];
        xi=[];
        b=0;
        
    end
    methods
        function obj= SMOAlgo(dataset)
            DsS=dataset;
            b=0;
            xi=DsS(:,1);
            yi=DsS(:,2);
            
            %initlializing the alpha such that yi.?i=0
            NNVals=sum(yi<0);
            NPVals=sum(yi>0);
            alpha1=randi([1 10],1,NPVals);
            alpha2=randi([-10 -1],1,NNVals);
            
            
            totalsum= sum(alpha1)+sum(alpha2);
            if(totalsum>0)
                alpha2(1,NNVals) = alpha2(1,NNVals)-totalsum;
            elseif (totalsum<0)
                alpha1(1,NPVals) = alpha1(1,NPVals)-totalsum;
            end
            
            al1=1,al2=1;
         for i=1:length(yi)
          if(yi(i,1)>0)
                 Alpha(1,i)=alpha1(1,al1);
                 al1=al1+1;
             else
                 Alpha(1,i)=-alpha2(1,al2);
                 al2=al2+1;
             end 
         end
         
         if(sum(Alpha*yi)==0)
            fprintf('yi.?i constraint statisfied'); 
         end
         
        end 
        
        
    end
    
    
end
