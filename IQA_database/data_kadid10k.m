function data_kadid10k(num_selection)
rng(0);
Dir = './kadid10k';
fileID = fopen(fullfile(Dir, 'dmos.csv'));
data = textscan(fileID,'%s %s %f %f\n', 'HeaderLines',1,'Delimiter',',');

imagename = data(:,1);
refnames_all = data(:,2);
mos_original = data(:,3);
std_original = data(:,4);

imagename = imagename{1,1};
refnames_all = refnames_all{1,1};
mos_original = mos_original{1,1};
std_original = std_original{1,1};

mos_original = mos_original';
std_original = sqrt(std_original');

mos = zeros(1,10125,'single');
std = zeros(1,10125,'single');
for i = 1:10125
    mos(1,i) = single(mos_original(i));
    std(1,i) = single(std_original(i));
end


refname = refnames_all(1:125:end);

for split = 1:10
    sel = randperm(81);
    train_path = [];
    train_mos = [];
    train_std = [];
    for i = 1:65
        train_sel = strcmpi(refname(sel(i)),refnames_all );
        train_sel = find(train_sel == 1);
        train_path = [train_path, imagename(train_sel)']; 
        train_mos = [train_mos,mos_original(train_sel)];
        train_std = [train_std,std_original(train_sel)];
    end

    test_path = [];
    test_mos = [];
    test_std = [];
    for i = 66:81
        test_sel = strcmpi(refname(sel(i)),refnames_all );
        test_sel = find(test_sel == 1);
        test_path = [test_path, imagename(test_sel)']; 
        test_mos = [test_mos,mos_original(test_sel)];
        test_std = [test_std,std_original(test_sel)];
    end
     
    %for train split
    train_index = 1:length(train_mos);
    %all_combination = nchoosek(train_index,2); %全组�?
    all_combination = comb(length(train_index));
    num_combines = size(all_combination);
    selected_index = randperm(num_combines(1));
    selected_index = selected_index(1:num_selection);
    combination = all_combination(selected_index,:);
    %combination = all_combination(selected_index,:);
    %combination = combination(1:150*2:end,:);
    
    fid = fopen(fullfile('./kadid10k/splits2/',num2str(split),'kadid10k_train.txt'),'w');
    for i = 1:length(combination)
        path1_index = combination(i,1);
        path2_index = combination(i,2);
        path1 = fullfile('images',train_path(path1_index));
        path1 = strrep(path1,'\','/');
        path1_mos = train_mos(path1_index);
        path1_std = train_std(path1_index);
        path2 = fullfile('images',train_path(path2_index));
        path2 = strrep(path2,'\','/');
        path2_mos = train_mos(path2_index);
        path2_std = train_std(path2_index);
        y = GT_Gaussian(path1_mos, path2_mos, path1_std, path2_std);
        if path1_mos > path2_mos
            yb = 1;
        else
            yb = 0;
        end
        %fprintf(fid,'%s\t%s\t%f\r',path1{1},path2{1},y);
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1{1},path2{1},y, path1_std, path2_std, yb);
    end
    fclose(fid);
    
       %for train_score split
    fid = fopen(fullfile('./kadid10k/splits2',num2str(split),'kadid10k_train_score.txt'),'w');
    for i = 1:length(train_path)
        path = fullfile('images',train_path(i));
        path = strrep(path,'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path{1},train_mos(i),train_std(i));
    end
    fclose(fid);
    
    %for test split
    fid = fopen(fullfile('./kadid10k/splits2',num2str(split),'kadid10k_test.txt'),'w');
    for i = 1:length(test_path)
        path = fullfile('images',test_path(i));
        path = strrep(path,'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path{1},test_mos(i),test_std(i));
    end
    fclose(fid);   
end

disp('kadid10k completed!');
