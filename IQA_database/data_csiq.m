function data_csiq(num_selection)
rng(0);
Dir = './CSIQ';

fileID = fopen(fullfile(Dir,'csiq.beta.txt'));
data = textscan(fileID,'%s	%d	%s	%d	%f	%f');
suffix = 'png';

refpath = fullfile(Dir,'src_imgs');
refpath = strcat(refpath,'/');
dir_rf = dir([refpath '*.png']);
imdb.refname = cell(1,30);
% dir_rf = dir(['dataset\CSIQ database\src_imgs\' '*.png']);
for i = 1:30
    file_name = dir_rf(i).name;
    %imdb.refname{i} = string(file_name);
    imdb.refname{i} = file_name(1:end-4);
end

imdb.refnames_all = data{1,1}';
imgpath = cell(1,length(data{1,1}));
dmos = zeros(1,length(data{1,1}));
std = zeros(1,length(data{1,1}));

AWGN_Index = 1;
BLUR_Index = 1;
contrast_Index = 1;
fnoise_Index = 1;
JPEG_Index = 1;
JPEG2000_Index = 1;

imdb.AWGN_path = cell(1,150);
imdb.BLUR_path = cell(1,150);
imdb.fnoise_path = cell(1,150);
imdb.JPEG_path = cell(1,150);
imdb.JPEG2000_path = cell(1,150);
imdb.contrast_path = cell(1,116);

imdb.AWGN_dmos = zeros(1,150);
imdb.BLUR_dmos = zeros(1,150);
imdb.fnoise_dmos = zeros(1,150);
imdb.JPEG_dmos = zeros(1,150);
imdb.JPEG2000_dmos = zeros(1,150);
imdb.contrast_dmos = zeros(1,116);

imdb.AWGN_std = zeros(1,150);
imdb.BLUR_std = zeros(1,150);
imdb.fnoise_std = zeros(1,150);
imdb.JPEG_std = zeros(1,150);
imdb.JPEG2000_std = zeros(1,150);
imdb.contrast_std = zeros(1,116);

imdb.AWGN_refname = cell(1,150);
imdb.BLUR_refname = cell(1,150);
imdb.fnoise_refname = cell(1,150);
imdb.JPEG_refname = cell(1,150);
imdb.JPEG2000_refname = cell(1,150);
imdb.contrast_refname = cell(1,116);


for i = 1:length(data{1,1})
    picname = data{1,1}(i);
    switch data{1,2}(i)
        case 1
            prefix = 'AWGN';
%             path =  'dataset\CSIQ database\dst_imgs\awgn\';
            path = 'dst_imgs\awgn';
%             path = fullfile(Dir,'dst_imgs\awgn');
            filename = strcat(picname,'.',prefix,'.',num2str(data{1,4}(i)),'.',suffix);
            filename = filename{1,1};
            imgpath{1,i} = fullfile(path,filename);
            dmos(i) = data{1,6}(i);
            std(i) = data{1,5}(i);
            
            imdb.AWGN_path{AWGN_Index} = imgpath{1,i};
            imdb.AWGN_dmos(AWGN_Index) = dmos(i);
            imdb.AWGN_std(AWGN_Index) = std(i);
            imdb.AWGN_refname{AWGN_Index} = data{1,1}(i);
            AWGN_Index = AWGN_Index + 1;
       case 2
            prefix = 'JPEG';
%             path = 'dataset\CSIQ database\dst_imgs\jpeg\';
%             path = fullfile(Dir,'dst_imgs\jpeg');
            path = 'dst_imgs\jpeg';
            filename = strcat(picname,'.',prefix,'.',num2str(data{1,4}(i)),'.',suffix);
            filename = filename{1,1};
            imgpath{1,i} = fullfile(path,filename);
            dmos(i) = data{1,6}(i);
            std(i) = data{1,5}(i);
            
            imdb.JPEG_path{JPEG_Index} = imgpath{1,i};
            imdb.JPEG_dmos(JPEG_Index) = dmos(i);
            imdb.JPEG_std(JPEG_Index) = std(i);
            imdb.JPEG_refname{JPEG_Index} = data{1,1}(i);
            JPEG_Index = JPEG_Index + 1;
        case 3
            prefix = 'jpeg2000';
%             path =  'dataset\CSIQ database\dst_imgs\jpeg2000\';
%             path = fullfile(Dir,'dst_imgs\jpeg2000');
            path = 'dst_imgs\jpeg2000';
            filename = strcat(picname,'.',prefix,'.',num2str(data{1,4}(i)),'.',suffix);
            filename = filename{1,1};
            imgpath{1,i} = fullfile(path,filename);
            dmos(i) = data{1,6}(i);
            std(i) = data{1,5}(i);
            
            imdb.JPEG2000_path{JPEG2000_Index} = imgpath{1,i};
            imdb.JPEG2000_dmos(JPEG2000_Index) = dmos(i);
            imdb.JPEG2000_std(JPEG2000_Index) = std(i);
            imdb.JPEG2000_refname{JPEG2000_Index} = data{1,1}(i);
            JPEG2000_Index = JPEG2000_Index + 1;
       case 4
            prefix = 'fnoise';
%             path =  'dataset\CSIQ database\dst_imgs\fnoise\';
%             path = fullfile(Dir,'dst_imgs\fnoise');
            path = 'dst_imgs\fnoise';
            filename = strcat(picname,'.',prefix,'.',num2str(data{1,4}(i)),'.',suffix);
            filename = filename{1,1};
            imgpath{1,i} = fullfile(path,filename);
            dmos(i) = data{1,6}(i);
            std(i) = data{1,5}(i);
            
            imdb.fnoise_path{fnoise_Index} = imgpath{1,i};
            imdb.fnoise_dmos(fnoise_Index) = dmos(i);
            imdb.fnoise_std(fnoise_Index) = std(i);
            imdb.fnoise_refname{fnoise_Index} = data{1,1}(i);
            fnoise_Index = fnoise_Index + 1;
        case 5
            prefix = 'BLUR';
%             path =  'dataset\CSIQ database\dst_imgs\blur\';
%             path = fullfile(Dir,'dst_imgs\blur');
            path = 'dst_imgs\blur';
            filename = strcat(picname,'.',prefix,'.',num2str(data{1,4}(i)),'.',suffix);
            filename = filename{1,1};
            imgpath{1,i} = fullfile(path,filename);
            dmos(i) = data{1,6}(i);
            std(i) = data{1,5}(i);
            
            imdb.BLUR_path{BLUR_Index} = imgpath{1,i};
            imdb.BLUR_dmos(BLUR_Index) = dmos(i);
            imdb.BLUR_std(BLUR_Index) = std(i);
            imdb.BLUR_refname{BLUR_Index} = data{1,1}(i);
            BLUR_Index = BLUR_Index + 1;
        case 6
            prefix = 'contrast';
%             path =  'dataset\CSIQ database\dst_imgs\contrast\';
%             path = fullfile(Dir,'dst_imgs\contrast');
            path = 'dst_imgs\contrast';
            filename = strcat(picname,'.',prefix,'.',num2str(data{1,4}(i)),'.',suffix);
            filename = filename{1,1};
            imgpath{1,i} = fullfile(path,filename);
            dmos(i) = data{1,6}(i);
            std(i) = data{1,5}(i);
            
            imdb.contrast_path{contrast_Index} = imgpath{1,i};
            imdb.contrast_dmos(contrast_Index) = dmos(i);
            imdb.contrast_std(contrast_Index) = std(i);
            imdb.contrast_refname{contrast_Index} = data{1,1}(i);
            contrast_Index = contrast_Index + 1;
    end  
end

imdb.imgpath = imgpath;
imdb.dmos = dmos;
imdb.std = std;
imdb.label = dmos;
imdb.filenum = 866;
imdb.refnum = 30;
imdb.dataset = 'CSIQ';


for split = 1:10
    sel = randperm(30);
    train_path = [];
    train_dmos = [];
    train_std = [];
    for i = 1:24
        train_sel = strcmpi(imdb.refname(sel(i)),imdb.refnames_all);
        train_sel = find(train_sel == 1);
        train_path = [train_path, imdb.imgpath(train_sel)]; 
        train_dmos = [train_dmos,imdb.label(train_sel)];
        train_std = [train_std,imdb.std(train_sel)];
    end

    test_path = [];
    test_dmos = [];
    test_std = [];
    for i = 25:30
        test_sel = strcmpi(imdb.refname(sel(i)),imdb.refnames_all);
        test_sel = find(test_sel == 1);
        test_path = [test_path, imdb.imgpath(test_sel)]; 
        test_dmos = [test_dmos,imdb.label(test_sel)];
        test_std = [test_std,imdb.std(test_sel)];
    end

    imdb.images.id = 1:866 ;
    imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
    imdb.images.label = [train_dmos,test_dmos];
    imdb.images.std = [train_std,test_std];
    imdb.classes.description = {'CSIQ'};
    imdb.images.name = [train_path,test_path] ;
    imdb.imageDir = Dir ;
    
    imdb.images.label = -imdb.images.label + 1; %调转单调�?
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
    %combination = combination(1:10:end,:);
    
    fid = fopen(fullfile('./CSIQ/splits2',num2str(split),'csiq_train.txt'),'w');
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
    fid = fopen(fullfile('./CSIQ/splits2',num2str(split),'csiq_train_score.txt'),'w');
    for i = 1:length(train_path)
        path = train_path(i);
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,train_dmos(i),train_std(i));
    end
    fclose(fid);
    
    
    %for test split
    fid = fopen(fullfile('./CSIQ/splits2',num2str(split),'csiq_test.txt'),'w');
    for i = 1:length(test_path)
        path = test_path(i);
        path = strrep(path{1,1},'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path,test_dmos(i),test_std(i));
    end
    fclose(fid);
end
fclose(fileID);
disp('csiq completed!');