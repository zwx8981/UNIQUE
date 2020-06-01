import torch
import os
import numpy as np

ckpt_dir = './checkpoint'
live_srcc = []
csiq_srcc = []
tid_srcc = []
bid_srcc = []
clive_srcc = []
koniq10k_srcc = []
kadid10k_srcc = []

index_cache = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total = 779 + 866 + 1162 + 586 + 10073 + 10125
weight = [779/total, 866/total, 1162/total, 586/total, 10073/total, 10125/total]

for i in range(0, 10):
    ckpt_session = os.path.join(ckpt_dir, str(i+1))
    ckpt = torch.load(os.path.join(ckpt_session, 'DataParallel-00011.pt'), map_location=device)

    live_results = ckpt['test_results_srcc']['live']
    csiq_results = ckpt['test_results_srcc']['csiq']
    tid_results = ckpt['test_results_srcc']['tid2013']
    bid_results = ckpt['test_results_srcc']['bid']
    clive_results = ckpt['test_results_srcc']['clive']
    koniq10k_results = ckpt['test_results_srcc']['koniq10k']
    kadid10k_results = ckpt['test_results_srcc']['kadid10k']


    weighted_results = np.array(live_results)*weight[0] + np.array(csiq_results)*weight[1] + \
                       np.array(bid_results)*weight[3] + np.array(clive_results)*weight[2] + \
                       np.array(koniq10k_results)*weight[4] + np.array(kadid10k_results)*weight[5]

    weighted_results = weighted_results.tolist()

    best_index = weighted_results.index(max(weighted_results))

    index_cache.append(best_index)

    live_srcc.append(live_results[best_index])
    csiq_srcc.append(csiq_results[best_index])
    tid_srcc.append(tid_results[best_index])
    bid_srcc.append(bid_results[best_index])
    clive_srcc.append(clive_results[best_index])
    koniq10k_srcc.append(koniq10k_results[best_index])
    kadid10k_srcc.append(kadid10k_results[best_index])

print('live median srcc: %f' % np.median(np.array(live_srcc)))
print('csiq median srcc: %f' % np.median(np.array(csiq_srcc)))
print('tid median srcc: %f' % np.median(np.array(tid_srcc)))
print('bid median srcc: %f' % np.median(np.array(bid_srcc)))
print('clive median srcc: %f' % np.median(np.array(clive_srcc)))
print('koniq10k median srcc: %f' % np.median(np.array(koniq10k_srcc)))
print('kadid10k median srcc: %f' % np.median(np.array(kadid10k_srcc)))
print(index_cache)

weighted_srcc = np.median(live_srcc) * weight[0] + np.median(csiq_srcc) * weight[1] + \
                   np.median(bid_srcc) * weight[3] + np.median(clive_srcc) * weight[2] + \
                   np.median(koniq10k_srcc) * weight[4] + np.median(kadid10k_srcc) * weight[5]

print('weighted_srcc: %f' % weighted_srcc)