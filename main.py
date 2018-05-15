# from models import *
import gc
import random
import numpy as np
import pandas as pd
# from functions import *
from models import *
import pickle
import glob
import re
import matplotlib.backends.backend_pdf
from functions import *





use_nn = False
use_shuff = False
use_shift = False
num_of_shifts = 70
use_voxelSearch = False
use_spatialAvg = False
stat_dict = {}

# stat_dict_rss = {}
batch_size = 8
num_of_electrodes = 136
for e in range(num_of_electrodes):
    stat_dict[e] = {}
electrodes = [1, 2, 3, 27, 28 ,29 ,30, 31, 32, 33, 34, 44, 45, 46, 47, 49,
              66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 103, 104, 114]
# electrodes = [66, 67, 68, 69, 70, 71, 72, 10, 11, 12, 50, 51, 52, 60]
# electrodes = [57, 66]

if use_voxelSearch:
    pdf = matplotlib.backends.backend_pdf.PdfPages("heatMaps_voxelSearch.pdf")
elif use_shuff:
    pdf = matplotlib.backends.backend_pdf.PdfPages("heatMaps_control.pdf")
else:
    pdf = matplotlib.backends.backend_pdf.PdfPages("heatMaps.pdf")



hfiles = glob.glob("/Users/berryweinstein/tensorflow/TF_FeatureExtraction/features_full*")
file_cnt = 0

state_saves = glob.glob("/Users/berryweinstein/PycharmProjects/ecog/venv/stats_save_*")

if state_saves:
    latest_save = max(state_saves, key=lambda p: int(re.findall(r'\d+', p)[0]))
    print("loading saved stat_dict from: %s" % (latest_save))
    stat_dict = pickle.load(open(latest_save, "rb"))
    file_cnt = len(stat_dict[0])
elif use_shift:
    print("loading clean saved stat_dict from: clean_total_stats.p")
    clean_stat_dict = pickle.load(open('clean_total_stats.p', "rb"))
    clean_layers = clean_stat_dict[0].keys()
timed_out_layers = []

# svd_save_dict = {}

for hf in hfiles:
    f = h5.File(hf, "r")
    dataset, layer = extract_last_dict_samples(f)

    if layer in stat_dict[0].keys():
        print ("Layer %s already exists, moving to next" % (layer))
        continue
    if use_shift and layer not in clean_layers:
        print("Layer %s wasn't proccesed in the clean run, moving to next" % (layer))
        continue

    if layer in timed_out_layers:
        print ("Layer %s already timed out, moving to next" % (layer))
        continue

    print ('Processing layer %s' % (layer))
    total_minibatches, layer = Data.create_train_test_data_for_reg(dataset, layer)

    # svd_save_dict[layer] = {}
    corr_arr1 = []

    X = [i[0] for i in total_minibatches]
    if use_nn:
        Y = [i[1][:, :] for i in total_minibatches] # TODO: fix according to the for loop
    else:
        Y = [i[1] for i in total_minibatches]
    if (use_shuff):
        random.shuffle(Y)
    if (use_shift):
        n = random.randint(int(len(total_minibatches) / 4), int(3 * len(total_minibatches) / 4))
        Y_tag = [i[1] for i in total_minibatches]
        Y = []
        # Y = np.array([i for i in Y_tag[n: len(Y_tag)]] + [i for i in Y_tag[0: n]])
        for j in range(num_of_shifts):
            n = random.randint(int(len(total_minibatches) / 4), int(3 * len(total_minibatches) / 4))
            Y += [l for l in Y_tag[n: len(Y_tag)]] + [k for k in Y_tag[0: n]]
            # Y = np.append(Y, [i for i in Y_tag[n: len(Y_tag)]] + [i for i in Y_tag[0: n]], axis=2)
        Y = np.array(Y).reshape([int(len(Y) / num_of_shifts), num_of_shifts * Y[0].shape[0] * Y[0].shape[1]])
    else:
        Y = np.array(Y).reshape([len(Y), Y[0].shape[0] * Y[0].shape[1]])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, shuffle=False)

    if use_nn:
        per_elec_output = forward_backward_net(X_train, X_test, y_train, y_test)
        per_elec_target = []
        for s in y_test:
            per_elec_target.extend(list(s))
        y_test = per_elec_target
    else:
        # alpha = cross_validate_alpha(np.array(X_train), np.array(y_train), layer, win)
        # alphas_dict[e][layer].append(alpha)
        X_train = np.array(X_train)
        X_valid = np.array(X_valid)
        X_test = np.array(X_test)
        # _, corr, _, _, _ = bootstrap_ridge(X_train, X_test, np.array(y_train), np.array(y_test),
        #                                                       alphas=np.logspace(-2, 2, 20),
        #                                                       nboots=5,
        #                                                       chunklen=10, nchunks=500)

        gen_alphas = np.logspace(0, 3, 20)
        corr, U, S, Vh = ridge_corr(X_train, X_valid, y_train, y_valid,
                          alphas=gen_alphas, return_svd=True)

    if corr == None:
        timed_out_layers.append(layer)
        continue

    # svd_save_dict[layer]['U'] = U
    # svd_save_dict[layer]['S'] = S
    # svd_save_dict[layer]['Vh'] = Vh

    corr = np.array(corr)

    aa = [np.argmax(k) for k in corr.T]
    # alphas_max = np.array([alphas[k] for k in aa])
    # w = ridge(X_test, y_test, alphas_max)
    corr_test = ridge_corr(X_train, X_test, y_train, y_test,
                                alphas=gen_alphas, U=U, S=S, Vh=Vh)
    corr_test = np.array(corr_test)
    corr_final = [c[aa[idx]] for idx, c in enumerate(corr_test.T)]
    corr_final = np.array(corr_final).reshape([num_of_electrodes, int(len(corr_final) / num_of_electrodes)])

    gc.collect()

    for idx, e_corr in enumerate(corr_final):
        if use_shift:
            stat_dict[idx][layer] = np.max(e_corr)
        else:
            stat_dict[idx][layer] = e_corr

    if use_shift:
        pickle.dump(stat_dict, open("stats_save_null_%d.p" % (file_cnt), "wb"))
    else:
        pickle.dump(stat_dict, open("stats_save_%d.p" % (file_cnt), "wb"))
    # pickle.dump(svd_save_dict, open("svd_save_dict_layer_%s.p" % (layer), "wb"))
    file_cnt += 1

    del total_minibatches
    gc.collect()

if use_shift:
    pickle.dump(stat_dict, open("total_stats_null.p", "wb"))
else:
    pickle.dump(stat_dict, open("total_stats.p", "wb"))
del stat_dict
gc.collect()




stat_dict = pickle.load(open("total_stats.p", 'rb'))
cstats = {}
final_stats = {}
for e in range(num_of_electrodes):
    cstats[e] = {}
    final_stats[e] = {}
for e in stat_dict:
    for k, v in stat_dict[e].items():
        if k == 'global_pool_':
            name = 'gp_last'
        else:
            ind = re.findall(r'block(\d)_unit_(\d)_bottleneck_v1_conv(\d)_', k)
            name = 'b' + str(ind[0][0]) + 'u' + str(ind[0][1]) + 'c' + str(ind[0][2])
        cstats[e][name] = v
    for key in sorted(cstats[e]):
        final_stats[e][key] = cstats[e][key]


with PdfPages('heatMaps.pdf') as pdf:
    for e in final_stats.keys():
        fig = create_heat_map(pd.DataFrame.from_dict(final_stats[e]).T, e)
        pdf.savefig(fig)
    pdf.close()






# if use_voxelSearch:
#     pickle.dump(stat_dict, open("stat_dict_pearson_voxelSearch.p", "wb"))
# else:
#     pickle.dump(stat_dict, open("stat_dict_pearson.p", "wb"))
# pickle.dump(stat_dict_rss, open("stat_dict_rss.p", "wb"))


        # for voxel in range(total_minibatches[0][0].shape[0]):
        #     reg = linear_model.Ridge(alpha=.5)
        #     X = [i[0][voxel] for i in total_minibatches]
        #     random.shuffle(total_minibatches)
        #     Y_shuff = [i[1] for i in total_minibatches]
        #     X_train, X_test, y_train, y_test = train_test_split(X, Y_shuff, test_size=0.2, shuffle=False)
        #     reg.fit(X_train, y_train)
        #     per_elec_output = reg.predict(X_test)
        #     stat_dict[e].append(stats.pearsonr(per_elec_output, y_test))






# print('valid loss: {:5.10f}'.format(sum_loss / n_valid))
#
#
# print("Pearson correleation (r, p)")
# print(
#
# print("Visual electrode 100 samples. Target in blue. Net output in green")
# plt.plot(per_elec_target[0:100])
# plt.plot(per_elec_output[0:100])
# plt.show()

# model_parameters = filter(lambda p: p.requires_grad, net.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
