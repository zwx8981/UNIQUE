function data_clive(num_selection)
rng(0);
Dir = './ChallengeDB_release';

imdb.imgpath = cell(1,1162);
imgpath = fullfile(Dir,'Data','AllImages_release.mat');
img = load(imgpath);
img = img.AllImages_release;

mospath = fullfile(Dir,'Data','AllMOS_release.mat');
mos = load(mospath);
mos = mos.AllMOS_release;
imdb.mos = mos(8:end);

stdpath = fullfile(Dir,'Data','AllStdDev_release.mat');
std = load(stdpath);
std = std.AllStdDev_release;
imdb.std = std(8:end);

for i = 8:1169
    file_name = img{i,1};     
    imdb.imgpath{i-7} = fullfile('Images',file_name);
end

for split = 1:10
    sel = randperm(1162);
    train_sel = sel(1:round(0.8*1162));
    test_sel = sel(round(0.8*1162)+1:end);

    train_path = imdb.imgpath(train_sel);
    test_path = imdb.imgpath(test_sel);

    train_mos = imdb.mos(train_sel);
    test_mos = imdb.mos(test_sel);
    
    train_std = imdb.std(train_sel);
    test_std = imdb.std(test_sel);

    imdb.images.id = 1:1162 ;
    imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
    imdb.images.label = [train_mos,test_mos];
    imdb.images.std = [train_std,test_std];

    imdb.classes.description = {'LIVE_CHAN'};
    imdb.images.name = [train_path,test_path] ;
    imdb.imageDir = Dir ;
    
    %for train split
    train_index = 1:length(train_mos);
    all_combination = nchoosek(train_index,2); %全组�?
    num_combines = size(all_combination);
    selected_index = randperm(num_combines(1));
    selected_index = selected_index(1:num_selection);
    combination = all_combination(selected_index,:);
    %combination = all_combination(selected_index,:);
    %combination = combination(1:20:end,:);
    
    fid = fopen(fullfile('./ChallengeDB_release/splits2',num2str(split),'clive_train.txt'),'w');
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
    fid = fopen(fullfile('./ChallengeDB_release/splits2',num2str(split),'clive_train_score.txt'),'w');
    for i = 1:length(train_path)
        path = train_path(i);
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,train_mos(i),train_std(i));
    end
    fclose(fid);
    
    %for test split
    fid = fopen(fullfile('./ChallengeDB_release/splits2',num2str(split),'clive_test.txt'),'w');
    for i = 1:length(test_path)
        path = test_path(i);
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,test_mos(i),test_std(i));
    end
    fclose(fid);

end

disp('clive completed!');
