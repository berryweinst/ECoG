import pickle
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
# from models import *
import h5py
import signal
import timeout_decorator
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Turn interactive plotting off
plt.ioff()



def extract_last_dict_samples(obj):
    next_key = ''
    if isinstance(obj, h5py.Group):
        next_key = list(obj.keys())[-1]
        return extract_last_dict_samples(obj[next_key])[0], next_key + '_' + extract_last_dict_samples(obj[next_key])[1]
    else:
        return obj, next_key

def plot_time_locked_avg():
    tl_avg = pickle.load(open('elec_tl_avg.p', 'rb'))
    with PdfPages('tl_elec.pdf') as pdf:
        for e in tl_avg:
            plt.figure(figsize=(3, 3))
            x1, x2, y1, y2 = plt.axis()
            plt.axis((0, 500, -2, 2))
            plt.plot(e)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


def plot_time_locked_avg_per_elec(e, ax):
    tl_avg = pickle.load(open('elec_tl_avg.p', 'rb'))
    x1, x2, y1, y2 = ax.axis()
    ax.axis((0, 500, -2, 2))
    ax0.plot(e)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()


def calc_r_squared(predicted_values, actual_values):
    rss = sum((np.array(predicted_values) - np.array(actual_values)) ** 2)
    tss = sum((np.array(actual_values) - np.mean(actual_values)) ** 2)
    return 1 - (rss/tss)




def create_heat_map(df, e):
    tl_avg = pickle.load(open('elec_tl_avg.p', 'rb'))
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(7, 9.6))
    # plot the heatmap
    ax0.set_title('Electrode number %d' % (e))
    hmap = sns.heatmap(df, vmin=-0.25, vmax=0.25)
    hmap.set_yticklabels(hmap.get_yticklabels(), rotation = 0, fontsize = 6)
    hmap.set_xticklabels(hmap.get_xticklabels(), rotation = 0, fontsize = 6)
    x1, x2, y1, y2 = ax0.axis()
    ax0.axis((0, 500, -2, 2))
    ax0.plot(tl_avg[e])
    # plt.close()
    return fig



def cross_validate_alpha(x, y, layer, win):
    print('Ridge Regression cross validation start on layer %s window %d' % (layer, win))
    print('alpha\t r value\n')
    alphas = np.linspace(.01, 1, 10)
    # cor_list = []
    val_scored = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        score = np.sum(cross_val_score(ridge,
                                         x,
                                         y=y,
                                         scoring='r2',
                                         cv=3,
                                         n_jobs=-1))
        val_scored.append(score)


    # kf = KFold(n_splits=10, shuffle=False)
    # for a in alpha:
    #     ridge = Ridge(fit_intercept=True, alpha=a)
    #     # reg_out = ridge
    #     # ridge.fit(x, y)
    #     # p = ridge.predict(x)
    #     # cor_out = stats.pearsonr(p, y)[0]
    #     cor = 0
    #     for train, test in kf.split(x, y):
    #         ridge.fit(x[train], y[train])
    #         p = ridge.predict(x[test])
    #         cor += stats.pearsonr(p, y[test])[0]
    #     cor /= 10
    #     cor_list.append(cor)
        print('{:.3f}\t {:.4f}'.format(alpha, score))
    a_max = alphas[np.argmax(val_scored)]
    return a_max



def forward_backward_net(X_train, X_test, y_train, y_test):
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, shuffle=False)

    net = Fetures2ECoGTrans(features_dim=X_test[0].shape, hidden_dim=int(X_test[0].shape[3] / 8))
    opt = torch.optim.RMSprop(net.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()

    n_train = len(y_train)
    epoch = 0
    hit_loss_grow = 0
    max_epochs = 50
    min_loss = np.inf
    n_valid = len(y_valid)
    while epoch < max_epochs:
        sum_loss = 0
        net.train()
        for idx, t in enumerate(X_train):
            net.zero_grad()
            w, output = net(Variable(t))
            loss = mse(output, Variable(y_train[idx]))
            loss.backward()
            opt.step()
            sum_loss += loss.data[0]
        epoch += 1
        print('[{:2d}] {:5.10f}'.format(epoch, sum_loss / n_train))
        sum_loss = 0
        net.eval()
        for idx, t in enumerate(X_valid):
            w, output = net(Volatile(t))
            loss = mse(output, Volatile(y_valid[idx]))
            sum_loss += loss.data[0]

        valid_loss = sum_loss / n_valid
        print('valid loss: {:5.10f}'.format(valid_loss))
        if (min_loss < valid_loss):
            hit_loss_grow += 1
            if (hit_loss_grow >= 3):
                break
        min_loss = valid_loss

    per_elec_output = []
    for idx, t in enumerate(X_test):
        w, output = net(Volatile(t))
        per_elec_output.extend(list(output.data))
    return per_elec_output






