function data_live(num_selection)
rng(0);
Dir = './databaserelease2';

refpath = fullfile(Dir,'refimgs');
refpath = strcat(refpath,'/');
dir_rf = dir([refpath '*.bmp']);
dmos_t = load(fullfile(Dir,'dmos_realigned.mat'));
imdb.dmos = dmos_t.dmos_new;
imdb.orgs = dmos_t.orgs;
imdb.std = dmos_t.dmos_std;

refname = load(fullfile(Dir,'refnames_all.mat'));
imdb.refnames_all = refname.refnames_all;

imdb.j2dmos = imdb.dmos(1:227);
imdb.jpdmos = imdb.dmos(228:460);
imdb.wndmos = imdb.dmos(461:634);
imdb.gbdmos = imdb.dmos(635:808);
imdb.ffdmos = imdb.dmos(809:end);

imdb.j2std = imdb.std(1:227);
imdb.jpstd = imdb.std(228:460);
imdb.wnstd = imdb.std(461:634);
imdb.gbstd = imdb.std(635:808);
imdb.ffstd = imdb.std(809:end);

imdb.j2orgs = imdb.orgs(1:227);
imdb.jporgs = imdb.orgs(228:460);
imdb.wnorgs = imdb.orgs(461:634);
imdb.gborgs = imdb.orgs(635:808);
imdb.fforgs = imdb.orgs(809:end);

imdb.orgs = [imdb.j2orgs,imdb.jporgs,imdb.wnorgs,imdb.gborgs,imdb.fforgs];

imdb.refname = cell(1,29);
for i = 1:29
    file_name = dir_rf(i).name;
    %imdb.refname{i} = string(file_name);
    imdb.refname{i} = file_name;
end


%%jp2k
index = 1;
imdb.dir_j2 = cell(1,227);
for i = 1:227
    file_name = strcat('img',num2str(i),'.bmp');
    imdb.dir_j2{index} = fullfile('jp2k',file_name);
    index = index + 1;
end

%%jpeg
index = 1;
imdb.dir_jp = cell(1,233);
for i = 1:233
    file_name = strcat('img',num2str(i),'.bmp');
    imdb.dir_jp{index} = fullfile('jpeg',file_name);
    index = index + 1;
end

%%white noise
index = 1;
imdb.dir_wn = cell(1,174);
for i = 1:174
       file_name = strcat('img',num2str(i),'.bmp');
       imdb.dir_wn{index} = fullfile('wn',file_name);
       index = index + 1;
end

%%gblur
index = 1;
imdb.dir_gb = cell(1,174);
for i = 1:174
    file_name = strcat('img',num2str(i),'.bmp');
    imdb.dir_gb{index} = fullfile('gblur',file_name);
    index = index + 1;
end

%%fast fading
index = 1;
imdb.dir_ff = cell(1,174);
for i = 1:174
    file_name = strcat('img',num2str(i),'.bmp');
    imdb.dir_ff{index} = fullfile('fastfading',file_name);
    index = index + 1;
end

imdb.imgpath =  cat(2,imdb.dir_j2,imdb.dir_jp,imdb.dir_wn,imdb.dir_gb,imdb.dir_ff);
imdb.dataset = 'LIVE';
imdb.filenum = 982;

for split = 1:10
    sel = randperm(29);
    train_path = [];
    train_dmos = [];
    train_std = [];
    for i = 1:23
        train_sel = strcmpi(imdb.refname(sel(i)),refname.refnames_all);
        train_sel = train_sel.*(~imdb.orgs);
        train_sel = find(train_sel == 1);
        train_path = [train_path, imdb.imgpath(train_sel)]; 
        train_dmos = [train_dmos,imdb.dmos(train_sel)];
        train_std = [train_std,imdb.std(train_sel)];
    end

    test_path = [];
    test_dmos = [];
    test_std = [];
    for i = 24:29
        test_sel = strcmpi(imdb.refname(sel(i)),refname.refnames_all);
        test_sel = test_sel.*(~imdb.orgs);
        test_sel = find(test_sel == 1);
        test_path = [test_path, imdb.imgpath(test_sel)]; 
        test_dmos = [test_dmos,imdb.dmos(test_sel)];
        test_std = [test_std,imdb.std(test_sel)];
    end
    
    imdb.images.id = 1:779 ;
    imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
    imdb.images.label = [train_dmos,test_dmos];
    imdb.images.std = [train_std,test_std];
    imdb.classes.description = {'LIVE'};
    imdb.images.name = [train_path,test_path] ;

    imdb.images.label = -imdb.images.label + max(imdb.images.label); %调转单调�?
    train_length = length(train_dmos);
    train_dmos = imdb.images.label(1:train_length);
    test_dmos = imdb.images.label(train_length+1:end);
    
    train_std = imdb.images.std(1:train_length);
    test_std = imdb.images.std(train_length+1:end);
    
    
    %for train split
    train_index = 1:length(train_dmos);
    all_combination = nchoosek(train_index,2); %全组�?
    num_combines = size(all_combination);
    selected_index = randperm(num_combines(1));
    selected_index = selected_index(1:num_selection);
    combination = all_combination(selected_index,:);
    %combination = all_combination(selected_index,:);
    %combination = combination(1:15:end,:);
    
    fid = fopen(fullfile('./databaserelease2/splits2',num2str(split),'live_train.txt'),'w');
    for i = 1:length(combination)
        path1_index = combination(i,1);
        path2_index = combination(i,2);
        path1 = train_path(path1_index);
        path1 = strrep(path1,'\','/');
        path1_dmos = train_dmos(path1_index);
        path1_std = train_std(path1_index);
        path2 = train_path(path2_index);
        path2 = strrep(path2,'\','/');
        path2_dmos = train_dmos(path2_index);
        path2_std = train_std(path2_index);
        y = GT_Gaussian(path1_dmos, path2_dmos, path1_std, path2_std);
        if path1_dmos > path2_dmos
            yb = 1;
        else
            yb = 0;
        end
        %fprintf(fid,'%s\t%s\t%f\r',path1{1},path2{1},y);
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1{1},path2{1},y, path1_std, path2_std,yb);
    end
    fclose(fid);
    
    %for train_score split
    fid = fopen(fullfile('./databaserelease2/splits2',num2str(split),'live_train_score.txt'),'w');
    for i = 1:length(train_path)
        path = train_path(i);
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,train_dmos(i),train_std(i));
    end
    fclose(fid);
    
    %for test split
    fid = fopen(fullfile('./databaserelease2/splits2',num2str(split),'live_test.txt'),'w');
    for i = 1:length(test_path)
        path = test_path(i);
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,test_dmos(i),test_std(i));
    end
    fclose(fid);
end
disp('live completed!');
