function labels_gaussian = get_gaussian_labels_negative(labels,sigma,x_grid,y_grid)
% labels_discrete = round(labels./grid_size);
% n_xlabels = length(unique(labels_discrete(:,1)));
% n_ylabels =length(unique(labels_discrete(:,2)));
n_xlabels=length(x_grid);
n_ylabels=length(y_grid);
if(iscolumn(x_grid))
    x_grid=x_grid';
end
if(isrow(y_grid))
    y_grid=y_grid';
end
labels_gaussian = zeros(size(labels,1),n_ylabels,n_xlabels);
map_X = repmat(x_grid,n_ylabels,1);
map_Y = repmat(y_grid,1,n_xlabels);
n_points = size(labels,1);
    for i=1:n_points
        d = (map_X-labels(i,1)).^2+(map_Y-labels(i,2)).^2;
        cur_gaussian = exp(-d/sigma/sigma);%*1/sqrt(2*pi)/output_sigma;        
        labels_gaussian(i,:)=cur_gaussian(:);
    end
end