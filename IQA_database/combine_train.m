live_root = 'databaserelease2';
csiq_root = 'CSIQ';
tid_root = 'TID2013';
kadid_root = 'kadid10k';

bid_root = 'BID';
clive_root = 'ChallengeDB_release';
koniq_root = 'koniq-10k';

for session = 1:10
    
    filename = fullfile(live_root,'splits2',num2str(session),'live_train.txt');
    fid = fopen(filename);
    live_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    filename = fullfile(csiq_root,'splits2',num2str(session),'csiq_train.txt');
    fid = fopen(filename);
    csiq_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    filename = fullfile(tid_root,'splits2',num2str(session),'tid_train.txt');
    fid = fopen(filename);
    tid_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);

    filename = fullfile(kadid_root,'splits2',num2str(session),'kadid10k_train.txt');
    fid = fopen(filename);
    kadid_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    filename = fullfile(clive_root,'splits2',num2str(session),'clive_train.txt');
    fid = fopen(filename);
    clive_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    filename = fullfile(bid_root,'splits2',num2str(session),'bid_train.txt');
    fid = fopen(filename);
    bid_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    filename = fullfile(koniq_root,'splits2',num2str(session),'koniq10k_train.txt');
    fid = fopen(filename);
    koniq_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    fid = fopen(fullfile('./splits2',num2str(session),'train.txt'),'w');
    %live
    for i = 1:length(live_data{1,1})
        path1 = live_data(1);
        path2 = live_data(2);
        y = live_data(3);
        std1 = live_data(4);
        std2 = live_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = live_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(live_root,path1{i,1});
        path2 = fullfile(live_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1),std1(i,1),std2(i,1),yb(i,1));
    end
%     %csiq
    for i = 1:length(csiq_data{1,1})
        path1 = csiq_data(1);
        path2 = csiq_data(2);
        y = csiq_data(3);
        std1 = csiq_data(4);
        std2 = csiq_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = csiq_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(csiq_root,path1{i,1});
        path2 = fullfile(csiq_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
    %tid
%     for i = 1:length(tid_data{1,1})
%         path1 = tid_data(1);
%         path2 = tid_data(2);
%         y = tid_data(3);
%         std1 = tid_data(4);
%         std2 = tid_data(5);
%         path1 = path1{1,1};
%         path2 = path2{1,1};
%         y = y{1,1};
%         std1 = std1{1,1};
%         std2 = std2{1,1};
%         path1 = fullfile(tid_root,path1{i,1});
%         path2 = fullfile(tid_root,path2{i,1});
%         path1 = strrep(path1, '\', '/');
%         path2 = strrep(path2, '\', '/');
%         fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
%     end
    %kadid
    for i = 1:length(kadid_data{1,1})
        path1 = kadid_data(1);
        path2 = kadid_data(2);
        y = kadid_data(3);
        std1 = kadid_data(4);
        std2 = kadid_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = kadid_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(kadid_root,path1{i,1});
        path2 = fullfile(kadid_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
%     %clive
    for i = 1:length(clive_data{1,1})
        path1 = clive_data(1);
        path2 = clive_data(2);
        y = clive_data(3);
        std1 = clive_data(4);
        std2 = clive_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = clive_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(clive_root,path1{i,1});
        path2 = fullfile(clive_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
   % bid
    for i = 1:length(bid_data{1,1})
        path1 = bid_data(1);
        path2 = bid_data(2);
        y = bid_data(3);
        std1 = bid_data(4);
        std2 = bid_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = bid_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(bid_root,path1{i,1});
        path2 = fullfile(bid_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
    %koniq10k
    for i = 1:length(koniq_data{1,1})
        path1 = koniq_data(1);
        path2 = koniq_data(2);
        y = koniq_data(3);
        std1 = koniq_data(4);
        std2 = koniq_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = koniq_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(koniq_root,path1{i,1});
        path2 = fullfile(koniq_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
    fclose(fid);
end

split = 1;

