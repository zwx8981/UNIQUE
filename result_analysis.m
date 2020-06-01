clear; close all; clc;
live_srcc = zeros(1,10);
live_plcc = zeros(1,10);
csiq_srcc = zeros(1,10);
csiq_plcc = zeros(1,10);
kadid10k_srcc = zeros(1,10);
kadid10k_plcc = zeros(1,10);
clive_srcc = zeros(1,10);
clive_plcc = zeros(1,10);
bid_srcc = zeros(1,10);
bid_plcc = zeros(1,10);
koniq10k_srcc = zeros(1,10);
koniq10k_plcc = zeros(1,10);

for i = 1:10
    result = load(fullfile('scores', strcat('scores',num2str(i),'.mat')));
    %live
    live_gmos = result.mos.live;
    live_pmos = result.hat.live;
    
%     live_gmos = -live_gmos + max(live_gmos);
%     live_pmos = -live_pmos + max(live_pmos);
    
    live_gstd = result.std.live;
    live_pstd = result.pstd.live;
    [live_srcc(i),~,live_plcc(i),~] = verify_performance(live_gmos,live_pmos);
%     [~,live_index] = sort(live_pmos);
%     ppc = ones(length(live_pmos),1);
%     figure(i); scatter(live_pmos, ppc); hold on
 
    %csiq
    csiq_gmos = result.mos.csiq;
    csiq_pmos = result.hat.csiq;
    
%     csiq_gmos = -csiq_gmos + max(csiq_gmos);
%     csiq_pmos = -csiq_pmos + max(csiq_pmos);
    
    csiq_gstd = result.std.csiq;
    csiq_pstd = result.pstd.csiq;
    [csiq_srcc(i),~,csiq_plcc(i),~] = verify_performance(csiq_gmos,csiq_pmos);
%     [~,csiq_index] = sort(csiq_pmos);
%     ppc = 2*ones(length(csiq_pmos),1);
%     figure(i); scatter(csiq_pmos, ppc); hold on

    %kadid10k
    kadid10k_gmos = result.mos.kadid10k;
    kadid10k_pmos = result.hat.kadid10k;
    kadid10k_gstd = result.std.kadid10k;
    kadid10k_pstd = result.pstd.kadid10k;
    [kadid10k_srcc(i),~,kadid10k_plcc(i),~] = verify_performance(kadid10k_gmos,kadid10k_pmos);
%     [~,kadid10k_index] = sort(kadid10k_pmos);
%     ppc = 3*ones(length(kadid10k_pmos),1);
%     figure(i); scatter(kadid10k_pmos, ppc); hold on

    %bid
    bid_gmos = result.mos.bid;
    bid_pmos = result.hat.bid;
    bid_gstd = result.std.bid;
    bid_pstd = result.pstd.bid;
    [bid_srcc(i),~,bid_plcc(i),~] = verify_performance(bid_gmos,bid_pmos);
%     [~,bid_index] = sort(bid_pmos);
%     ppc = 4*ones(length(bid_pmos),1);
%     figure(i); scatter(bid_pmos, ppc); hold on

    %clive
    clive_gmos = result.mos.clive;
    clive_pmos = result.hat.clive;
    clive_gstd = result.std.clive;
    clive_pstd = result.pstd.clive;
    [clive_srcc(i),~,clive_plcc(i),~] = verify_performance(clive_gmos,clive_pmos);
%    [~,clive_index] = sort(clive_pmos);
%     ppc = 5*ones(length(clive_pmos),1);
%     figure(i); scatter(clive_pmos, ppc);

    %koniq10k
    koniq10k_gmos = result.mos.koniq10k;
    koniq10k_pmos = result.hat.koniq10k;
    koniq10k_gstd = result.std.koniq10k;
    koniq10k_pstd = result.pstd.koniq10k;
    [koniq10k_srcc(i),~,koniq10k_plcc(i),~] = verify_performance(koniq10k_gmos,koniq10k_pmos);
    
%     allpmos = [clive_pmos',live_pmos',csiq_pmos',koniq10k_pmos',kadid10k_pmos',bid_pmos'];
%     allpstd = [clive_pstd',live_pstd',csiq_pstd',koniq10k_pstd',kadid10k_pstd',bid_pstd'];
%     
%     synpmos = [live_pmos',csiq_pmos',kadid10k_pmos'];
%     synpstd = [live_pstd',csiq_pstd',kadid10k_pstd'];
%     
%     realpmos = [clive_pmos',koniq10k_pmos',bid_pmos'];
%     realpstd = [clive_pstd',koniq10k_pstd',bid_pstd'];
    
%     scatter(live_pmos, live_pstd); hold on;
%     scatter(csiq_pmos, csiq_pstd); hold on;
%     scatter(kadid10k_pmos, kadid10k_pstd); hold on;
%     scatter(bid_pmos, bid_pstd); hold on;
%     scatter(clive_pmos, clive_pstd); hold on;
%     scatter(koniq10k_pmos, koniq10k_pstd);

%     [~,koniq10k_index] = sort(koniq10k_pmos);
%     ppc = 6*ones(length(koniq10k_pmos),1);
%     figure(i); scatter(koniq10k_pmos, ppc);
    
%     figure(1); scatter(live_gmos, live_gstd,'MarkerEdgeColor', [185,141,122]/255, 'MarkerFaceColor',[185,141,122]/255); 
%     set(gca,'LooseInset',get(gca,'TightInset'));
%     scatter(live_gmos, live_gstd,'MarkerEdgeColor', [185,141,122]/255, 'MarkerFaceColor',[185,141,122]/255);grid on; hold on;
%     xlabel('DMOS','FontName','Times New Roman','FontSize',60);
%     ylabel('Predicted std','FontName','Times New Roman','FontSize',60);
    figure(2); scatter(live_pmos, live_pstd,'MarkerEdgeColor', [185,141,122]/255, 'MarkerFaceColor',[185,141,122]/255);
    set(gca,'LooseInset',get(gca,'TightInset'));
    scatter(live_pmos, live_pstd,'MarkerEdgeColor', [185,141,122]/255, 'MarkerFaceColor',[185,141,122]/255);grid on;hold on;
    set(gca,'FontName','Times New Roman','FontSize',50);
    xlabel('$f_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    ylabel('$\sigma_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
  %  set(gca,'YTick',0.9:0.02:1.1);
    %%%
%     figure(3); scatter(csiq_gmos, csiq_gstd,'MarkerEdgeColor', [250,210,63]/255, 'MarkerFaceColor',[250,210,63]/255); 
%     set(gca,'LooseInset',get(gca,'TightInset'));
%     scatter(csiq_gmos, csiq_gstd, 'MarkerEdgeColor', [250,210,63]/255, 'MarkerFaceColor',[250,210,63]/255);grid on;hold on;
%     xlabel('DMOS','FontName','Times New Roman','FontSize',60);
%     ylabel('Predicted std','FontName','Times New Roman','FontSize',60);
    figure(4); scatter(csiq_pmos, csiq_pstd, 'MarkerEdgeColor', [250,210,63]/255, 'MarkerFaceColor',[250,210,63]/255);
    set(gca,'LooseInset',get(gca,'TightInset'));
    scatter(csiq_pmos, csiq_pstd, 'MarkerEdgeColor', [250,210,63]/255, 'MarkerFaceColor',[250,210,63]/255);grid on;hold on;
    set(gca,'FontName','Times New Roman','FontSize',50);
    xlabel('$f_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    ylabel('$\sigma_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
   % set(gca,'YTick',0.9:0.02:1.1);
    %%%
%     figure(5); scatter(kadid10k_gmos, kadid10k_gstd,'MarkerEdgeColor', [222,91,21]/255, 'MarkerFaceColor',[222,91,21]/255);
%     set(gca,'LooseInset',get(gca,'TightInset'));
%     scatter(kadid10k_gmos, kadid10k_gstd,'MarkerEdgeColor', [222,91,21]/255, 'MarkerFaceColor',[222,91,21]/255);grid on;hold on;
%     xlabel('Predicted MOS','FontName','Times New Roman','FontSize',60);
%     ylabel('Predicted std','FontName','Times New Roman','FontSize',60);
    figure(6); scatter(kadid10k_pmos, kadid10k_pstd,'MarkerEdgeColor', [222,91,21]/255, 'MarkerFaceColor',[222,91,21]/255);
    set(gca,'LooseInset',get(gca,'TightInset'));
    scatter(kadid10k_pmos, kadid10k_pstd, 'MarkerEdgeColor',[222,91,21]/255, 'MarkerFaceColor',[222,91,21]/255);grid on;hold on;
    set(gca,'FontName','Times New Roman','FontSize',50);
    xlabel('$f_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    ylabel('$\sigma_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    %set(gca,'YTick',0.9:0.02:1.1);
    %%%
%     figure(7); scatter(bid_gmos, bid_gstd, 'MarkerEdgeColor',[133,122,175]/255, 'MarkerFaceColor',[133,122,175]/255);
%     set(gca,'LooseInset',get(gca,'TightInset'));
%     scatter(bid_gmos, bid_gstd,'MarkerEdgeColor', [133,122,175]/255, 'MarkerFaceColor',[133,122,175]/255);grid on;hold on;
%     xlabel('Predicted MOS','FontName','Times New Roman','FontSize',60);
%     ylabel('Predicted std','FontName','Times New Roman','FontSize',60);
    figure(8); scatter(bid_pmos, bid_pstd,'MarkerEdgeColor', [133,122,175]/255, 'MarkerFaceColor',[133,122,175]/255);
    set(gca,'LooseInset',get(gca,'TightInset'));
    scatter(bid_pmos, bid_pstd, 'MarkerEdgeColor',[133,122,175]/255, 'MarkerFaceColor',[133,122,175]/255);grid on;hold on;
    set(gca,'FontName','Times New Roman','FontSize',50);
    xlabel('$f_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    ylabel('$\sigma_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    %set(gca,'YTick',0.9:0.02:1.1);
    %%%    
%     figure(9); scatter(clive_gmos, clive_gstd, 'MarkerEdgeColor',[78,33,53]/255, 'MarkerFaceColor',[78,33,53]/255);
%     set(gca,'LooseInset',get(gca,'TightInset'));
%     scatter(clive_gmos, clive_gstd, 'MarkerEdgeColor',[78,33,53]/255, 'MarkerFaceColor',[78,33,53]/255);grid on;hold on;
%     xlabel('Predicted MOS','FontName','Times New Roman','FontSize',60);
%     ylabel('Predicted std','FontName','Times New Roman','FontSize',60);
    figure(10); scatter(clive_pmos, clive_pstd,'MarkerEdgeColor', [78,33,53]/255, 'MarkerFaceColor',[78,33,53]/255);
    set(gca,'LooseInset',get(gca,'TightInset'));
    scatter(clive_pmos, clive_pstd, 'MarkerEdgeColor',[78,33,53]/255, 'MarkerFaceColor',[78,33,53]/255);grid on;hold on;
    set(gca,'FontName','Times New Roman','FontSize',50);
    xlabel('$f_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    ylabel('$\sigma_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    %set(gca,'YTick',0.9:0.02:1.1);
    %%%
%     figure(11); scatter(koniq10k_gmos, koniq10k_gstd, 'MarkerEdgeColor',[255,65,82]/255, 'MarkerFaceColor',[255,65,82]/255);
%     set(gca,'LooseInset',get(gca,'TightInset'));
%     scatter(koniq10k_gmos, koniq10k_gstd,'MarkerEdgeColor', [255,65,82]/255, 'MarkerFaceColor',[255,65,82]/255);grid on;hold on;
%     xlabel('Predicted MOS','FontName','Times New Roman','FontSize',60);
%     ylabel('Predicted std','FontName','Times New Roman','FontSize',60);
    figure(12); scatter(koniq10k_pmos, koniq10k_pstd,'MarkerEdgeColor', [255,65,82]/255, 'MarkerFaceColor',[255,65,82]/255);
    set(gca,'LooseInset',get(gca,'TightInset'));
    scatter(koniq10k_pmos, koniq10k_pstd, 'MarkerEdgeColor',[255,65,82]/255, 'MarkerFaceColor',[255,65,82]/255);grid on;hold on;
    set(gca,'FontName','Times New Roman','FontSize',50);
    xlabel('$f_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    ylabel('$\sigma_{w}(x)$','interpreter','latex','FontSize',50,'FontName','Times New Roman');
    %set(gca,'YTick',0.9:0.02:1.1);
end

live_srcc = median(live_srcc); live_plcc = median(live_plcc);
csiq_srcc = median(csiq_srcc); csiq_plcc = median(csiq_plcc);
kadid10k_srcc = median(kadid10k_srcc); kadid10k_plcc = median(kadid10k_plcc);
bid_srcc = median(bid_srcc); bid_plcc = median(bid_plcc);
clive_srcc = median(clive_srcc); clive_plcc = median(clive_plcc);
koniq10k_srcc = median(koniq10k_srcc); koniq10k_plcc = median(koniq10k_plcc);

weighted_srcc = live_srcc*779 + csiq_srcc*866 + kadid10k_srcc*10125 + bid_srcc*586 + clive_srcc*1162 + koniq10k_srcc*10073 ;
weighted_plcc = live_plcc*779 + csiq_plcc*866 + kadid10k_plcc*10125 + bid_plcc*586 + clive_plcc*1162 + koniq10k_plcc*10073 ;
weighted_srcc = weighted_srcc / (779+866+10125+586+1162+10073);
weighted_plcc = weighted_plcc / (779+866+10125+586+1162+10073);