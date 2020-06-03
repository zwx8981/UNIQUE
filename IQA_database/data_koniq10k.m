function data_koniq10k(num_selection)
rng(0);
Dir = './koniq-10k';
load(fullfile(Dir,'koniq10k_scores_and_distributions.mat'));
data = koniq10kscoresanddistributions;
%data = csvread(fullfile(Dir,'koniq10k_scores_and_distributions.csv'));

%num = 10073
imagename = data(:,1);
mos_original = data(:,8);
std_original = data(:,9);
mos = zeros(1,10073,'single');
std = zeros(1,10073,'single');
for i = 1:10073
    mos(1,i) = single(str2double(mos_original(i)));
    std(1,i) = single(str2double(std_original(i)));
end
for split = 1:10
    sel = randperm(10073);
    train_sel = sel(1:round(0.8*10073));
    test_sel = sel(round(0.8*10073)+1:end);
    train_path = imagename(train_sel);
    test_path = imagename(test_sel);
    train_mos = mos(train_sel);
    test_mos = mos(test_sel);
    train_std = std(train_sel);
    test_std = std(test_sel);
     
    %for train split
    train_index = 1:length(train_mos);
    %all_combination = nchoosek(train_index,2); %全组�?
    all_combination = comb(length(train_sel));
    num_combines = size(all_combination);
    selected_index = randperm(num_combines(1));
    selected_index = selected_index(1:num_selection);
    combination = all_combination(selected_index,:);
    %combination = all_combination(selected_index,:);
    %combination = combination(1:150*2:end,:);
    
    fid = fopen(fullfile('./koniq-10k/splits2/',num2str(split),'koniq10k_train.txt'),'w');
    for i = 1:length(combination)
        path1_index = combination(i,1);
        path2_index = combination(i,2);
        path1 = fullfile('1024x768',train_path(path1_index));
        path1 = strrep(path1,'\','/');
        path1_mos = train_mos(path1_index);
        path1_std = train_std(path1_index);
        path2 = fullfile('1024x768',train_path(path2_index));
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
    fid = fopen(fullfile('./koniq-10k/splits2',num2str(split),'koniq10k_train_score.txt'),'w');
    for i = 1:length(train_path)
        path = fullfile('1024x768',train_path(i));
        path = strrep(path,'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,train_mos(i),train_std(i));
    end
    fclose(fid);
    
    %for test split
    fid = fopen(fullfile('./koniq-10k/splits2',num2str(split),'koniq10k_test.txt'),'w');
    for i = 1:length(test_path)
        path = fullfile('1024x768',test_path(i));
        path = strrep(path,'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,test_mos(i),test_std(i));
    end
    fclose(fid);   
end

disp('koniq10k completed!');
