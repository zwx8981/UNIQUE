scores_root = './scores';

live_floss = zeros(1,10);
csiq_floss = zeros(1,10);
kadid10k_floss = zeros(1,10);
bid_floss = zeros(1,10);
clive_floss = zeros(1,10);
koniq10k_floss = zeros(1,10);

for session = 1:10
    filename = strcat('scores',num2str(session),'.mat');
    scores_path = fullfile(scores_root, filename);
    scores = load(scores_path);
    %live
    live_moss = scores.mos.live;
    live_stds = scores.std.live;
    live_hats = scores.hat.live;
    live_pstds = scores.pstd.live;        
    
    num = length(live_moss);
    b = nchoosek(1:num,2);
    floss = 0;
    for j = 1:size(b,1)
        gmos1 = live_moss(b(j,1));
        gstd1 = live_stds(b(j,1));
        gmos2 = live_moss(b(j,2));
        gstd2 = live_stds(b(j,2));
        q = GT_Gaussian(gmos1, gmos2, gstd1, gstd2);
        
        pmos1 = live_hats(b(j,1));
        pstd1 = live_pstds(b(j,1));
        pmos2 = live_hats(b(j,2));
        pstd2 = live_pstds(b(j,2));
        p = GT_Gaussian(pmos1, pmos2, pstd1, pstd2);
        
        floss = floss + fidelity_loss(p, q);
    end
    
    live_floss(session) = floss/size(b,1);
    
    %csiq
    csiq_moss = scores.mos.csiq;
    csiq_stds = scores.std.csiq;
    csiq_hats = scores.hat.csiq;
    csiq_pstds = scores.pstd.csiq;
    
    num = length(csiq_moss);
    b = nchoosek(1:num,2);
    floss = 0;
    for j = 1:size(b,1)
        gmos1 = csiq_moss(b(j,1));
        gstd1 = csiq_stds(b(j,1));
        gmos2 = csiq_moss(b(j,2));
        gstd2 = csiq_stds(b(j,2));
        q = GT_Gaussian(gmos1, gmos2, gstd1, gstd2);
        
        pmos1 = csiq_hats(b(j,1));
        pstd1 = csiq_pstds(b(j,1));
        pmos2 = csiq_hats(b(j,2));
        pstd2 = csiq_pstds(b(j,2));
        p = GT_Gaussian(pmos1, pmos2, pstd1, pstd2);
        
        floss = floss + fidelity_loss(p, q);
    end
    
    csiq_floss(session) = floss/size(b,1);
    
    %kadid10k
    kadid10k_moss = scores.mos.kadid10k;
    kadid10k_stds = scores.std.kadid10k;
    kadid10k_hats = scores.hat.kadid10k;
    kadid10k_pstds = scores.pstd.kadid10k;
    
    num = length(kadid10k_moss);
    b = nchoosek(1:num,2);
    floss = 0;
    for j = 1:size(b,1)
        gmos1 = kadid10k_moss(b(j,1));
        gstd1 = kadid10k_stds(b(j,1));
        gmos2 = kadid10k_moss(b(j,2));
        gstd2 = kadid10k_stds(b(j,2));
        q = GT_Gaussian(gmos1, gmos2, gstd1, gstd2);
        
        pmos1 = kadid10k_hats(b(j,1));
        pstd1 = kadid10k_pstds(b(j,1));
        pmos2 = kadid10k_hats(b(j,2));
        pstd2 = kadid10k_pstds(b(j,2));
        p = GT_Gaussian(pmos1, pmos2, pstd1, pstd2);
        
        floss = floss + fidelity_loss(p, q);
    end
    
    kadid10k_floss(session) = floss/size(b,1);
    
    %bid
    bid_moss = scores.mos.bid;
    bid_stds = scores.std.bid;
    bid_hats = scores.hat.bid;
    bid_pstds = scores.pstd.bid;
    
    num = length(bid_moss);
    b = nchoosek(1:num,2);
    floss = 0;
    for j = 1:size(b,1)
        gmos1 = bid_moss(b(j,1));
        gstd1 = bid_stds(b(j,1));
        gmos2 = bid_moss(b(j,2));
        gstd2 = bid_stds(b(j,2));
        q = GT_Gaussian(gmos1, gmos2, gstd1, gstd2);
        
        pmos1 = bid_hats(b(j,1));
        pstd1 = bid_pstds(b(j,1));
        pmos2 = bid_hats(b(j,2));
        pstd2 = bid_pstds(b(j,2));
        p = GT_Gaussian(pmos1, pmos2, pstd1, pstd2);
        
        floss = floss + fidelity_loss(p, q);
    end
    
    bid_floss(session) = floss/size(b,1);
    
    %clive
    clive_moss = scores.mos.clive;
    clive_stds = scores.std.clive;
    clive_hats = scores.hat.clive;
    clive_pstds = scores.pstd.clive;
    
    num = length(clive_moss);
    b = nchoosek(1:num,2);
    floss = 0;
    for j = 1:size(b,1)
        gmos1 = clive_moss(b(j,1));
        gstd1 = clive_stds(b(j,1));
        gmos2 = clive_moss(b(j,2));
        gstd2 = clive_stds(b(j,2));
        q = GT_Gaussian(gmos1, gmos2, gstd1, gstd2);
        
        pmos1 = clive_hats(b(j,1));
        pstd1 = clive_pstds(b(j,1));
        pmos2 = clive_hats(b(j,2));
        pstd2 = clive_pstds(b(j,2));
        p = GT_Gaussian(pmos1, pmos2, pstd1, pstd2);
        
        floss = floss + fidelity_loss(p, q);
    end
    
    clive_floss(session) = floss/size(b,1);
    
    
    %clive
    koniq10k_moss = scores.mos.koniq10k;
    koniq10k_stds = scores.std.koniq10k;
    koniq10k_hats = scores.hat.koniq10k;
    koniq10k_pstds = scores.pstd.koniq10k;
    
    num = length(koniq10k_moss);
    b = nchoosek(1:num,2);
    floss = 0;
    for j = 1:size(b,1)
        gmos1 = koniq10k_moss(b(j,1));
        gstd1 = koniq10k_stds(b(j,1));
        gmos2 = koniq10k_moss(b(j,2));
        gstd2 = koniq10k_stds(b(j,2));
        q = GT_Gaussian(gmos1, gmos2, gstd1, gstd2);
        
        pmos1 = koniq10k_hats(b(j,1));
        pstd1 = koniq10k_pstds(b(j,1));
        pmos2 = koniq10k_hats(b(j,2));
        pstd2 = koniq10k_pstds(b(j,2));
        p = GT_Gaussian(pmos1, pmos2, pstd1, pstd2);
        
        floss = floss + fidelity_loss(p, q);
    end
    
    koniq10k_floss(session) = floss/size(b,1);
    
    % for split = 1:10
    %     fid = fopen(fullfile(root,num2str(split),'live_test.txt'));
    %     data = textscan(fid, '%s %f %f');
    %     files = data{1,1}; gmoss = data{1,2}; gstds = data{1,3};
    %     num = length(gmoss);
    %     b = nchoosek(1:num,2);
    %     for j = 1:size(b,1)
    %         gmos1 = gmoss(b(j,1));
    %         gstd1 = gstds(b(j,1));
    %         gmos2 = gmoss(b(j,2));
    %         gstd2 = gstds(b(j,2));
    %         p = GT_Gaussian(gmos1, gmos2, gstd1, gstd2);
    %     end
    % end
    
end

zlive = median(live_floss);
zcsiq = median(csiq_floss);
zkadid10k = median(kadid10k_floss);
zbid = median(bid_floss);
zclive = median(clive_floss);
zkoniq10k = median(koniq10k_floss);

function fidelity = fidelity_loss(p, q)
fidelity = 1 - sqrt(p*q) - sqrt((1-p)*(1-q));
end


function p = GT_Gaussian(mos1, mos2, std1, std2)
p = (mos1 - mos2) / (sqrt(std1^2 + std2^2) + eps);
p = normcdf(p);
end