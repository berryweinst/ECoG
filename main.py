# from models import *
import gc
import random
import numpy as np
import pandas as pd
# from functions import *
from models import *
import pickle
import re
import matplotlib.backends.backend_pdf
from functions import *





use_nn = False
use_shuff = False
use_shift = True
num_of_shifts = 100
use_voxelSearch = False
use_spatialAvg = False
stat_dict = {}

# stat_dict_rss = {}
batch_size = 8
num_of_electrodes = 136
for e in range(num_of_electrodes):
    stat_dict[e] = {}
# electrodes = [1, 2, 3, 27, 28 ,29 ,30, 31, 32, 33, 34, 44, 45, 46, 47, 49, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 103, 104, 114]
# electrodes = [66, 67, 68, 69, 70, 71, 72, 10, 11, 12, 50, 51, 52, 60]
# electrodes = [57, 66]

if use_voxelSearch:
    pdf = matplotlib.backends.backend_pdf.PdfPages("heatMaps_voxelSearch.pdf")
elif use_shuff:
    pdf = matplotlib.backends.backend_pdf.PdfPages("heatMaps_control.pdf")
else:
    pdf = matplotlib.backends.backend_pdf.PdfPages("heatMaps.pdf")


# for e in electrodes:
#     print ("Training model on electrode number %d" % (e))
    # if use_nn:
    #     stat_dict[e] = {}
    #     layer_win_pairs = [(i, j) for i in ['block1', 'block2', 'block3', 'blocks4', 'conv1'] for j in range(45)]
    #     for layer, _ in layer_win_pairs:
    #         stat_dict[e][layer] = []
    #         # stat_dict_rss[e][layer] = []
    #     for layer, win in layer_win_pairs:
    #         print('Processing layer %s, window %d' % (layer, win))
    #
    #         total_minibatches = Data.create_minibatches('resnet_v1_50', layer, win, batch_size, e)
    #
    #         X_train, X_test, y_train, y_test = train_test_split([i[0] for i in total_minibatches],
    #                                                             [i[1] for i in total_minibatches],
    #                                                             test_size=0.2, shuffle=False)
    #
    #         X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, shuffle=False)
    #
    #
    #
    #         # net = Fetures2ECoGTrans(features_dim=X_test[0].shape[3], hidden_dim=y_test[0].shape[1])
    #         net = Fetures2ECoGTrans(features_dim=X_test[0].shape, hidden_dim=int(X_test[0].shape[3]/8))
    #         opt = torch.optim.RMSprop(net.parameters(), lr=1e-3)
    #         mse = torch.nn.MSELoss()
    #
    #         n_train = len(y_train)
    #         epoch = 0
    #         hit_loss_grow = 0
    #         max_epochs = 50
    #         min_loss = np.inf
    #         n_valid = len(y_valid)
    #         while epoch < max_epochs:
    #             sum_loss = 0
    #             for idx, t in enumerate(X_train):
    #                 net.zero_grad()
    #                 w, output = net(Variable(t))
    #                 loss = mse(output, Variable(y_train[idx]))
    #                 loss.backward()
    #                 opt.step()
    #                 sum_loss += loss.data[0]
    #             epoch += 1
    #             print('[{:2d}] {:5.10f}'.format(epoch, sum_loss / n_train))
    #             sum_loss = 0
    #             for idx, t in enumerate(X_valid):
    #                 w, output = net(Volatile(t))
    #                 loss = mse(output, Volatile(y_valid[idx]))
    #                 sum_loss += loss.data[0]
    #
    #             valid_loss = sum_loss / n_valid
    #             print('valid loss: {:5.10f}'.format(valid_loss))
    #             if (min_loss < valid_loss):
    #                 if (hit_loss_grow > 3):
    #                     break
    #                 hit_loss_grow += 1
    #             min_loss = valid_loss
    #
    #
    #
    #
    #         n_valid = len(y_test)
    #         sum_loss = 0
    #         # per_elec_target = np.empty((0, y_test[0].shape[1]))
    #         # per_elec_output = np.empty((0, y_test[0].shape[1]))
    #         per_elec_target = []
    #         per_elec_output = []
    #         for s in y_test:
    #             per_elec_target.extend(list(s))
    #             # per_elec_target = np.append(per_elec_target, np.array(s), axis=0)
    #
    #         for idx, t in enumerate(X_test):
    #             w, output = net(Volatile(t))
    #             per_elec_output.extend(list(output.data))
    #             loss = mse(output, Volatile(y_test[idx]))
    #             sum_loss += loss.data[0]
    #
    #
    #         stat_dict[e][layer].append(stats.pearsonr(per_elec_target, per_elec_output))
    #         # stat_dict[e].append(calc_r_squared(per_elec_output, per_elec_target))
    #         gc.collect()


        # SHUFFLE and rerun ###
        # print ("Shuffling and re trainning on electrode %d" %(e))
        # X = [i[0] for i in total_minibatches]
        # random.shuffle(total_minibatches)
        # Y_shuff = [i[1] for i in total_minibatches]
        # # n = random.randint(int(len(total_minibatches) / 4), int(3 * len(total_minibatches) / 4))
        # # Y_shuff = [i[1] for i in total_minibatches[n: len(total_minibatches)]] + [i[1] for i in total_minibatches[0: n]]
        # X_train, X_test, y_train, y_test = train_test_split(X, Y_shuff, test_size=0.2, shuffle=False)
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, shuffle=False)
        #
        # # net = Fetures2ECoGTrans(features_dim=X_test[0].shape[3], hidden_dim=y_test[0].shape[1])
        # net = Fetures2ECoGTrans(features_dim=X_test[0].shape, hidden_dim=int(X_test[0].shape[3] / 8))
        # opt = torch.optim.RMSprop(net.parameters(), lr=1e-3)
        # mse = torch.nn.MSELoss()
        #
        # n_train = len(y_train)
        # epoch = 0
        # hit_loss_grow = 0
        # max_epochs = 50
        # min_loss = np.inf
        # n_valid = len(y_valid)
        # while epoch < max_epochs:
        #     sum_loss = 0
        #     for idx, t in enumerate(X_train):
        #         net.zero_grad()
        #         w, output = net(Variable(t))
        #         loss = mse(output, Variable(y_train[idx]))
        #         loss.backward()
        #         opt.step()
        #         sum_loss += loss.data[0]
        #     epoch += 1
        #     print('[{:2d}] {:5.10f}'.format(epoch, sum_loss / n_train))
        #     sum_loss = 0
        #     for idx, t in enumerate(X_valid):
        #         w, output = net(Volatile(t))
        #         loss = mse(output, Volatile(y_valid[idx]))
        #         sum_loss += loss.data[0]
        #
        #     valid_loss = sum_loss / n_valid
        #     print('valid loss: {:5.10f}'.format(valid_loss))
        #     if (min_loss < valid_loss):
        #         if (hit_loss_grow > 3):
        #             break
        #         hit_loss_grow += 1
        #     min_loss = valid_loss
        #
        # n_valid = len(y_test)
        # sum_loss = 0
        # # per_elec_target = np.empty((0, y_test[0].shape[1]))
        # # per_elec_output = np.empty((0, y_test[0].shape[1]))
        # per_elec_target = []
        # per_elec_output = []
        # for s in y_test:
        #     per_elec_target.extend(list(s))
        #     # per_elec_target = np.append(per_elec_target, np.array(s), axis=0)
        #
        # for idx, t in enumerate(X_test):
        #     w, output = net(Volatile(t))
        #     per_elec_output.extend(list(output.data))
        #     loss = mse(output, Volatile(y_test[idx]))
        #     sum_loss += loss.data[0]
        #
        # # stat_dict[e].append(stats.pearsonr(per_elec_target, per_elec_output))
        # stat_dict[e].append(calc_r_squared(per_elec_output, per_elec_target))
        # gc.collect()
        #
        # print("Done electrode %d, Pearson statistics (1st - real, 2nd- shuffled)" % (e))
        # print(stat_dict[e])
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        #
        # N = len(electrodes)
        # real_stat = [v[0] for k, v in stat_dict.items()]
        # shuff_stat = [v[1] for k, v in stat_dict.items()]
        #
        # ind = np.arange(N)  # the x locations for the groups
        # width = 0.35  # the width of the bars
        #
        # ## the bars
        # real_bar = ax.bar(ind, real_stat, width,
        #                   color='blue')
        #
        # shuff_bar = ax.bar(ind + width, shuff_stat, width,
        #                    color='red')
        #
        # # axes and labels
        # ax.set_xlim(-width, len(ind) + width)
        # # ax.set_ylim(0,45)
        # ax.set_ylabel('R value')
        # ax.set_title('R per electrode - non-shuffled VS shuffled data')
        #
        # xTickMarks = [k for k in stat_dict.keys()]
        # ax.set_xticks(ind + width)
        # xtickNames = ax.set_xticklabels(xTickMarks)
        # plt.setp(xtickNames, rotation=45, fontsize=10)
        #
        # ## add a legend
        # ax.legend((real_bar[0], shuff_bar[0]), ('Non-shuffled', 'Shuffled'))
        #
        # plt.show()


    # else:
# stat_dict[e] = {}
# alphas_dict[e] = {}
# stat_dict_rss[e] = {}
# if use_voxelSearch:
#     total_minibatches = Data.create_train_test_data_for_reg('resnet_v1_50', e, 'voxel')
# elif use_spatialAvg:
#     total_minibatches = Data.create_train_test_data_for_reg('resnet_v1_50', e, 'avg')
# else:
#     exclude_layer = []
#     if use_nn:
#         print("Using NN to predict ECoG")
#         exclude_layer = ['conv1']
#         total_minibatches = Data.create_minibatches('resnet_v1_50', batch_size, e)
#         layer_win_pairs = [(i, j) for i in total_minibatches[0][0].keys() if i not in exclude_layer for j in
#                            range(total_minibatches[0][1].shape[1])]
#
#     else:
#         print("Using Ridge to predict ECoG")
#         total_minibatches = Data.create_train_test_data_for_reg('resnet_v1_50', 'full')
#         # layer_win_pairs = [(i, j) for i in total_minibatches[0][0].keys() if i not in exclude_layer for j in
#         #                    range(total_minibatches[0][1].shape[0])]
#         layers = [i for i in total_minibatches[0][0].keys() if i not in exclude_layer]

# for layer, _ in layer_win_pairs:
    # stat_dict[e][layer] = []
    # alphas_dict[e][layer] = []
    # stat_dict_rss[e][layer] = []
# alpha = None
import glob
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
# timed_out_layers = []

# svd_save_dict = {}

for hf in hfiles:
    total_minibatches, layer = Data.create_train_test_data_for_reg('resnet_v1_50', 'full', hf)
    # layers = [i for i in total_minibatches[0][0].keys()]

    # for layer in layers:

    if layer in stat_dict[0].keys():
        print ("Layer %s already exists, moving to next" % (layer))
        continue
    if use_shift and layer not in clean_layers:
        print("Layer %s wasn't proccesed in the clean run, moving to next" % (layer))
        continue

    # if layer in timed_out_layers:
    #     print ("Layer %s already timed out, moving to next" % (layer))
    #     continue
    print ('Processing layer %s' % (layer))
    # svd_save_dict[layer] = {}
    corr_arr1 = []
    # corr_arr2 = []
    # if use_voxelSearch:
    #     for voxel in range(total_minibatches[0][0][layer].shape[0]):
    #         X = [i[0][layer][voxel] for i in total_minibatches]
    #         Y = [i[1][win] for i in total_minibatches]
    #         if (use_shuff):
    #             random.shuffle(Y)
    #         if (use_shift):
    #             n = random.randint(int(len(total_minibatches) / 4), int(3 * len(total_minibatches) / 4))
    #             Y = [i for i in Y[n: len(Y)]] + [i for i in Y[0: n]]
    #
    #         X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    #         reg.fit(X_train, y_train)
    #         per_elec_output = reg.predict(X_test)
    #         corr_arr1.append(stats.pearsonr(per_elec_output, y_test))
    #         # corr_arr2.append(calc_r_squared(per_elec_output, y_test))
    # else:
    X = [i[0] for i in total_minibatches]
    if use_nn:
        Y = [i[1][:, :] for i in total_minibatches] # TODO: fix according to the for loop
    else:
        Y = [i[1] for i in total_minibatches]
    if (use_shuff):
        random.shuffle(Y)
    if (use_shift):
        n = random.randint(int(len(total_minibatches) / 4), int(3 * len(total_minibatches) / 4))
        Y = np.array([i for i in Y[n: len(Y)]] + [i for i in Y[0: n]])
        for i in range(num_of_shifts - 1):
            n = random.randint(int(len(total_minibatches) / 4), int(3 * len(total_minibatches) / 4))
            Y_tag = [i for i in Y[n: len(Y)]] + [i for i in Y[0: n]]
            Y = np.append(Y, Y_tag, axis=2)

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
        # alpha = .5
        # reg = CDRegressor(alpha=alpha)
        # reg.fit(np.array(X_train), np.array(y_train))
        # per_elec_output = reg.predict(X_test)

    # corr = stats.pearsonr(per_elec_output, y_test)
    # corr = reg.score(X_test, y_test)
    if corr == None:
        # timed_out_layers.append(layer)
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

# cstats = {}
# final_stats = {}
# for e in range(num_of_electrodes):
#     cstats[e] = {}
#     final_stats[e] = {}
# for e in stat_dict:
#     for k, v in stat_dict[e].items():
#         ind = re.findall(r'block(\d)_unit_(\d)_bottleneck_v1_conv(\d)_', k)
#         name = 'b' + str(ind[0][0]) + 'u' + str(ind[0][1]) + 'c' + str(ind[0][2])
#         cstats[e][name] = v
#     for key in sorted(cstats[e]):
#         final_stats[e][key] = cstats[e][key]
#
#
# with PdfPages('heatMaps.pdf') as pdf:
#     for e in final_stats.keys():
#         fig = create_heat_map(pd.DataFrame.from_dict(final_stats[e]).T, e)
#         pdf.savefig(fig)
#     pdf.close()
#
#
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
