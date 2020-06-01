function data_TID2013(num_selection)

rng(0);
Dir = './TID2013';

fileID = fopen(fullfile(Dir,'mos_with_names.txt'));
mos_name = textscan(fileID,'%f %s');
mos = mos_name{1,1};
name = mos_name{1,2};

fileID2 = fopen(fullfile(Dir,'mos_std.txt'));
mos_name2 = textscan(fileID2,'%f %s');
std = mos_name2{1,1};

path = cell(1,3000);
for i = 1:3000
    namet = name{i,1};
%     path{1,i} = fullfile(Dir,'distorted_images',namet); 
    path{1,i} = fullfile('distorted_images',namet); 
end

imdb.mos = mos';
imdb.std = std';
imdb.label = imdb.mos;
imdb.name = name;
imdb.imgpath = path;
imdb.filenum = 3000;
imdb.refnum = 25;
imdb.dataset = 'TID2013';

refpath = fullfile(Dir,'reference_images');
refpath = strcat(refpath,'/');
dir_rf = dir([refpath '*.BMP']);
imdb.refname = cell(1,25);
for i = 1:25
    file_name = dir_rf(i).name;
    %imdb.refname{i} = string(file_name);
    imdb.refname{i} = file_name;
end

imdb.refnames_all = cell(1,3000);
for i = 1:25
    for j = 1:120
        imdb.refnames_all{1,(i-1)*120+j} = imdb.refname{i};
    end
end


for split = 1:10
    sel = randperm(25);
    train_path = [];
    train_mos = [];
    train_std = [];
    for i = 1:20
        train_sel = strcmpi(imdb.refname(sel(i)),imdb.refnames_all );
        train_sel = find(train_sel == 1);
        train_path = [train_path, imdb.imgpath(train_sel)]; 
        train_mos = [train_mos,imdb.label(train_sel)];
        train_std = [train_std,imdb.std(train_sel)];
    end

    test_path = [];
    test_mos = [];
    test_std = [];
    for i = 21:25
        test_sel = strcmpi(imdb.refname(sel(i)),imdb.refnames_all );
        test_sel = find(test_sel == 1);
        test_path = [test_path, imdb.imgpath(test_sel)]; 
        test_mos = [test_mos,imdb.label(test_sel)];
        test_std = [test_std,imdb.std(test_sel)];
    end

    imdb.images.id = 1:3000 ;
    imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
    imdb.images.label = [train_mos,test_mos];
    imdb.images.std = [train_std,test_std];
    imdb.images.label = imdb.images.label;
    imdb.classes.description = {'TID2013'};
    imdb.images.name = [train_path,test_path] ;
    imdb.imageDir = Dir ;
    
    %for train split
    train_index = 1:length(train_mos);
    %all_combination = nchoosek(train_index,2); %全组�?
    all_combination = comb(length(train_index));
    num_combines = size(all_combination);
    selected_index = randperm(num_combines(1));
    selected_index = selected_index(1:num_selection);
    combination = all_combination(selected_index,:);
    %combination = all_combination(selected_index,:);
    %combination = combination(1:35:end,:);
    
    fid = fopen(fullfile('./TID2013/splits2',num2str(split),'tid_train.txt'),'w');
    for i = 1:length(combination)
        path1_index = combination(i,1);
        path2_index = combination(i,2);
        path1 = train_path(path1_index);
        path1 = strrep(path1,'\','/');
        path1_mos = train_mos(path1_index);
        path1_std = train_std(path1_index);
        path2 = train_path(path2_index);
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
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1{1},path2{1},y, path1_std, path2_std,yb);
    end
    fclose(fid);
    
     %for train_score split
    fid = fopen(fullfile('./TID2013/splits2',num2str(split),'tid_train_score.txt'),'w');
    for i = 1:length(train_path)
        path = train_path(i);
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,train_mos(i),train_std(i));
    end
    fclose(fid);
    
    %for test split
    fid = fopen(fullfile('./TID2013/splits2',num2str(split),'tid_test.txt'),'w');
    for i = 1:length(test_path)
        path = test_path(i);
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,test_mos(i),test_std(i));
    end
    fclose(fid);
    
end
fclose(fileID);
disp('TID2013 completed!');
