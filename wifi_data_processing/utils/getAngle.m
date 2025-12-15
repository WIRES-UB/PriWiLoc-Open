function T=getAngle(ap_pos,coord)
    %ap_pos: Position of antennas (antenna 1 comes first
    %format is n_ant X 2)
    %coord: x,y coordinates
    ap_vec = ap_pos(1,:)-ap_pos(end,:);
    ap_center=mean(ap_pos);
    X=coord(:,1)-ap_center(1);
    Y=coord(:,2)-ap_center(2);
    T=zeros(size(X));
    for i=1:length(X(:))
        T(i) = sign(sum([X(i),Y(i)].*ap_vec))*(pi/2-acos(abs(sum([X(i),Y(i)].*ap_vec))/norm([X(i),Y(i)])/norm(ap_vec)));
    end
end