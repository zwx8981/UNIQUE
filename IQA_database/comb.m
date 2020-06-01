function all_comb = comb(num)
n=num;
A=tril(ones(n,'uint8'));
ind=sub2ind([n n],1:n,1:n);
A(ind)=0;
[u,v]=find(A(1:round(n/2),:));
ind1=uint16([v u]);
clear u v ind
[u,v]=find(A(round(n/2)+1:n,:));
clear A
ind2=[uint16(v) uint16(u)+round(n/2)];
clear u v
ind=[ind1;ind2];
all_comb = ind;
clear ind1 ind2
end