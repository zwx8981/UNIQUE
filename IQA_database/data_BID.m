function data_BID(num_selection)
rng(0);
Dir = './BID';

data = xlsread(fullfile(Dir, 'DatabaseGrades.xls'));
name = data(:,1);
mos = data(:, 2);
grades = data(:, 4:end);
mosstd = std(grades, 0, 2, 'omitnan');

imdb.std = mosstd;
imdb.mos = mos;
imdb.name = name;

for i = 1:length(name)
    num = name(i);
    if num < 10
        prefix = 'DatabaseImage000';
    elseif num < 100
        prefix = 'DatabaseImage00';
    else
        prefix = 'DatabaseImage0';
    end
    name_t = strcat(prefix, num2str(num), '.JPG');
    all_name{i} = name_t;
end

imdb.imgpath = all_name;

for split = 1:10
    sel = randperm(586);
    train_sel = sel(1:round(0.8*586));
    test_sel = sel(round(0.8*586)+1:end);

    train_path = imdb.imgpath(train_sel);
    test_path = imdb.imgpath(test_sel);
    %train_path = imdb.name(train_sel);
    %test_path = imdb.name(test_sel);

    train_mos = imdb.mos(train_sel);
    test_mos = imdb.mos(test_sel);

    train_std = imdb.std(train_sel);
    test_std = imdb.std(test_sel);
    
    imdb.images.id = 1:586 ;
    imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
    imdb.images.label = [train_mos',test_mos'];
    imdb.images.std = [train_std',test_std'];

    imdb.classes.description = {'BID'};
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
    %combination = combination(1:15:end,:);
    
    fid = fopen(fullfile('./BID/splits2',num2str(split),'bid_train.txt'),'w');
    for i = 1:length(combination)
        path1_index = combination(i,1);
        path2_index = combination(i,2);
        path1 = fullfile('ImageDatabase',train_path(path1_index));
        path1 = strrep(path1,'\','/');
        path1_mos = train_mos(path1_index);
        path1_std = train_std(path1_index);
        path2 = fullfile('ImageDatabase',train_path(path2_index));
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
    fid = fopen(fullfile('./BID/splits2',num2str(split),'bid_train_score.txt'),'w');
    for i = 1:length(train_path)
        path = fullfile('ImageDatabase',train_path(i));
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,train_mos(i),train_std(i));
    end
    fclose(fid);
    
    %for test split
    fid = fopen(fullfile('./BID/splits2',num2str(split),'bid_test.txt'),'w');
    for i = 1:length(test_path)
        path = fullfile('ImageDatabase',test_path(i));
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,test_mos(i),test_std(i));
    end
    fclose(fid);
end

disp('BID completed!');
