# -*- coding: utf-8 -*-
#
# discription: define utility functions for HMM_tools

# --基本モジュール--
import numpy as np
import pandas as pd
import sys
import os
from scipy import linalg
import random

# ---時間関連---
import datetime
from timeout_decorator import timeout

# モデリング関連
from hmmlearn import hmm

# 最適化
import optimization as opt

# ログ用
from logging import getLogger, StreamHandler, FileHandler, INFO, ERROR
import traceback

# ----------グラフ関連----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager  # 日本語フォントの表示で利用

np.random.seed(0)
random.seed(0)

# ログ処理
cmd = "hmm_utils"
pid = os.getpid()
logfile = "/tmp/hmm_tools_"+str(pid)+".log"
logger = getLogger(cmd)
Fhandler = FileHandler(logfile)
Fhandler.setLevel(INFO)
logger.addHandler(Fhandler)
Shandler = StreamHandler()
Shandler.setLevel(ERROR)
logger.addHandler(Shandler)
logger.setLevel(INFO)


# ログ出力関数
def error_exit(msg):
    d = datetime.datetime.today()
    logger.error(d.strftime("%Y-%m-%d %H:%M:%S")+" ERROR "+cmd+" - "+str(msg))
    # 例外発生
    raise Exception


def warn_print(msg):
    d = datetime.datetime.today()
    logger.warn(d.strftime("%Y-%m-%d %H:%M:%S")+" WARN "+cmd+" - "+str(msg))


def debug_print(msg):
    d = datetime.datetime.today()
    logger.info(d.strftime("%Y-%m-%d %H:%M:%S")+" INFO "+cmd+" - "+str(msg))


# 流用関数
# http://mglab.blogspot.jp/2010/04/blog-post_9805.html
#
# i->jに遷移する確率を行列(i,j)で表現した遷移確率行列から、定常分布を求めます。
# この関数は行列を転置する必要がない点に注意してください。
#
# mat : N×Nの遷移確率行列(numpyのmatrixでなければならない)
# 戻り値 : N次元の定常分布ベクトル
#
def getFSS(mat, cmd_arg="nan"):
    la, v = linalg.eig(mat.T)
    result = zip(la, v.T)

    # 定常分布を求めるため、固有値が1に近い順で固有ベクトルをソートする
    result_sorted = sorted(result,
                           cmp=lambda x, y: cmp(abs(1.0-x), abs(1.0-y)),
                           key=lambda x: x[0])

    # 固有値が1近傍でなかったら例外を送出  # 元は、1.0e-6
    # eig_value = result_sorted[0][0]
    # assert abs(eig_value - 1.0) < 1.0e-4, "the eigen value of FSS is apart \
    #                                        from 1.0"
    if abs(result_sorted[0][0] - 1.0) > 1.0e-4:
        error_exit("the eigen value of FSS is apart from 1.0. [getFSS] "
                   + "command: "+str(cmd_arg))

    fss = result_sorted[0][1]
    return fss/sum(fss)  # 正規化


# 自作関数 Gaussian用
GA_pop = 500
GA_gen = 3
FUNCTION_TIMEOUT = 3600


# 目的：GaussianHMMのパラメータ推定（平均、分散、初期確率、推移行列）
# 引数：対象データ(配列)、HMMの状態数、初期値の最適化をするか、
#       返り値2に値を入れるか(システム化する際にはDFは見ないため、False)
# 返り値1：fitしたモデルオブジェクト
# 返り値2；パラメタ推定値を1行のDataFrameにしたもの(outDF=Trueの場合のみ)
def G_HMM_para_est(in_data, state_num=2, opt=True, outDF=False, cmd_arg="nan"):
    # hmmlearn用データ加工
    if len(np.shape(in_data)) == 1:
        in_data = np.c_[in_data]
    elif len(np.shape(in_data)) == 2:
        if np.shape(in_data)[1] == 1:
            in_data = np.c_[in_data]
        else:
            error_exit("shape of in_data is unexpected. np.shape(in_data): "
                       + str(np.shape(in_data))+" [G_HMM_para_est] command: "
                       + str(cmd_arg))
    else:
        error_exit("shape of in_data is unexpected. np.shape(in_data): "
                   + str(np.shape(in_data))+" [G_HMM_para_est] command: "
                   + str(cmd_arg))
    # モデルオブジェクト生成
    if opt is True:  # 初期値最適化あり
        hmm_obj = hmm.GaussianHMM(n_components=state_num,
                                  covariance_type="full",
                                  init_params="")
        debug_print("start G_HMM_optimization(). command: "+str(cmd_arg))
        hmm_obj.startprob_, hmm_obj.transmat_, hmm_obj.means_, \
            hmm_obj.covars_ = G_HMM_optimization(in_data, state_num,
                                                 cmd_arg=cmd_arg)
        debug_print("hmm_obj.startprob_:"
                    + str([str(x) for x in hmm_obj.startprob_])
                    + ", hmm_obj.transmat_:"
                    + str([str(x) for x in hmm_obj.transmat_])
                    + ", hmm_obj.means_:"
                    + str([str(x) for x in hmm_obj.means_])
                    + ", hmm_obj.covars_:"
                    + str([str(x) for x in hmm_obj.covars_]))
        debug_print("end G_HMM_optimization(). command: "+str(cmd_arg))
    else:  # なし
        hmm_obj = hmm.GaussianHMM(n_components=state_num,
                                  covariance_type="full")
        # 初期化が必要？　データの次元を知らせる n_features
        hmm_obj.means_ = np.array([[0]])

    # モデリング
    debug_print("start hmm_obj.fit(). command: "+str(cmd_arg))

    # fit関数実行にタイムアウト処理を追加するためのラッパー関数
    @timeout(FUNCTION_TIMEOUT)
    def wrapper_hmmfit(data):
        hmm_obj.fit(data)

    wrapper_hmmfit(in_data)
    debug_print("end hmm_obj.fit(). command: "+str(cmd_arg))

    # ValueError: rows of transmat_ must sum to 1.0 の対策でrowsの合計を1にする
    hmm_obj.transmat_ = (hmm_obj.transmat_.T/sum(hmm_obj.transmat_.T)).T

    # meanの小さい順に各状態のパラメータ値を入れ替える
    indexer = np.argsort(np.ravel(hmm_obj.means_))
    # hmm_obj.startprob_ = hmm_obj.startprob_[indexer]
    # 初期確率を全て同じ確率とする場合
    hmm_obj.startprob_ = list(np.repeat(float(1)/state_num, state_num))
    hmm_obj.transmat_ = hmm_obj.transmat_[indexer].T[indexer].T
    # 初期確率を定常確率とする場合
    # hmm_obj.startprob_ = getFSS(hmm_obj.transmat_)
    hmm_obj.means_ = hmm_obj.means_[indexer]
    hmm_obj.covars_ = hmm_obj.covars_[indexer]

    if outDF is False:
        # 空のDFを返す
        hmm_resDF = pd.DataFrame()
    else:
        state_list = np.arange(state_num)
        debug_print("start making result dataframe. command: "+str(cmd_arg))
        # 尤度関係
        # AIC=(-2)*LLH+2*param_num
        # BIC=(-2)+LLH+log(n)*param_num
        # パラメータ数(param_num)：
        #    分布のパラメータ数*状態数(K*s)+初期状態確率(s-1)+推移行列(s*(s-1))
        #    ⇒s^2 + ks - 1
        LLH = hmm_obj.score(in_data)
        # AIC = (-2)*LLH+2*(state_num**2+state_num*2-1)
        BIC = (-2)*LLH+(np.log(len(in_data)))*(state_num**2+state_num*2-1)
        # AIC_LLH = pd.DataFrame({"State_num": [state_num],
        #                       "aic": [AIC], "llh": [LLH]})
        BIC_LLH = pd.DataFrame({"State_num": [state_num],
                               "bic": [BIC], "llh": [LLH]})

        # パラメータ取得
        initProb = pd.DataFrame(np.ravel(hmm_obj.startprob_)).T
        initProb.columns = ["startprob_"+str(x) for x in state_list]
        transMat = pd.DataFrame(np.ravel(hmm_obj.transmat_)).T
        transMat.columns = ["transprob_"+str(x)+"_"+str(y)
                            for x in state_list for y in state_list]
        try:
            sProb = pd.DataFrame(np.ravel(getFSS(hmm_obj.transmat_,
                                                 cmd_arg=cmd_arg))).T
        except:
            sProb = pd.DataFrame(np.repeat(np.nan, state_num)).T
        sProb.columns = ["stationaryProb_"+str(x) for x in state_list]
        x_means = pd.DataFrame(np.ravel(hmm_obj.means_)).T
        x_means.columns = ["mean_"+str(x) for x in state_list]
        x_vars = pd.DataFrame(np.ravel(hmm_obj.covars_)).T
        x_vars.columns = ["var_"+str(x) for x in state_list]

        # 結合
        hmm_resDF = \
            pd.concat([BIC_LLH, initProb, transMat, sProb, x_means, x_vars],
                      axis=1)
        debug_print("end making result dataframe. command: "+str(cmd_arg))

    return hmm_obj, hmm_resDF


# 目的：GaussianHMMパラメータ推定で用いる初期値の最適化
# 引数：データ配列、HMMの状態数
# 返り値：各パラメータの初期値（初期状態確率、推移行列、平均、分散）
def G_HMM_optimization(data, ncompo, popsize=GA_pop, maxiter=GA_gen,
                       cmd_arg="nan"):
    # hmmlearn用データ加工
    if len(np.shape(data)) == 1:
        in_data = np.c_[data]
    elif len(np.shape(data)) == 2:
        if np.shape(data)[1] == 1:
            in_data = np.c_[data]
        else:
            error_exit("shape of in_data is unexpected. np.shape(data): "
                       + str(np.shape(data))+" [G_HMM_optimization]"
                       + " command: "+str(cmd_arg))
    else:
        error_exit("shape of in_data is unexpected. np.shape(data): "
                   + str(np.shape(data))+" [G_HMM_optimization]"
                   + " command: "+str(cmd_arg))

    # 最適化の領域定義
    dom_s = np.ravel([(1, 100)]*(ncompo))
    dom_t = np.ravel([(1, 100)]*(ncompo**2))
    dom_m = np.ravel([(int(min(in_data)*10), int(max(in_data)*10))]*ncompo)
    in_var = np.var(in_data)
    dom_c = \
        np.ravel([(max(1, int(in_var/10)), max(1, int(in_var*100)))]*ncompo)

    # (パラメータ数,上限下限(=2))でreshapeする
    domain = \
        np.hstack([dom_s, dom_t, dom_m, dom_c]).reshape((ncompo**2+3*ncompo),
                                                        2)

    # 目的関数定義 LLHを最大化する
    # opt.geneticoptimizeに渡す目的関数の引数は、最適化する配列の1つしか渡せない
    def llh(x):
        if isinstance(x, type(None)):
            warn_print("arg type is None [llh] command: "+str(cmd_arg))
            return np.inf
        hmm_obj = hmm.GaussianHMM(n_components=ncompo,
                                  covariance_type="full", init_params="")
        hmm_obj.startprob_, hmm_obj.transmat_, hmm_obj.means_, \
            hmm_obj.covars_ = G_HMM_conv_para(x, ncompo, cmd_arg=cmd_arg)

        # fit関数実行にタイムアウト処理を追加するためのラッパー関数
        @timeout(FUNCTION_TIMEOUT)
        def wrapper_hmmfit(data):
            hmm_obj.fit(data)

        try:
            wrapper_hmmfit(in_data)
        except Exception as exception:
            warn_print("function error. msg: "+str(exception)
                       + " opt_x:"+str(x)
                       + ", hmm_obj.startprob_:"
                       + str([str(val) for val in hmm_obj.startprob_])
                       + ", hmm_obj.transmat_:"
                       + str([str(val) for val in hmm_obj.transmat_])
                       + ", hmm_obj.means_:"
                       + str([str(val) for val in hmm_obj.means_])
                       + ", hmm_obj.covars_:"
                       + str([str(val) for val in hmm_obj.covars_])
                       + " [llh/hmmlearn.fit] command: "+str(cmd_arg))
            # fit()がエラーとなった場合は、infを返す
            return np.inf

        # fit()の結果、transmatの行または列が全て0になる場合もinfを返す
        trmat_DF = pd.DataFrame(hmm_obj.transmat_)
        if (trmat_DF.T.sum().min() == 0) | (trmat_DF.sum().min() == 0):
            warn_print("hmm_obj.transmat_ must sum to 1.0 ("
                       + str(np.array(trmat_DF.T.sum()))
                       + ","+str(np.array(trmat_DF.sum()))
                       + ") [llh] command: "+str(cmd_arg))
            return np.inf

        hmm_obj.transmat_ = (hmm_obj.transmat_.T/sum(hmm_obj.transmat_.T)).T

        # score関数実行にタイムアウト処理を追加するためのラッパー関数
        @timeout(FUNCTION_TIMEOUT)
        def wrapper_hmmscore(data):
            return hmm_obj.score(data)

        try:
            tmp_score = wrapper_hmmscore(in_data)*(-1)
        except Exception as exception:
            warn_print("function error. msg: "+str(exception)
                       + " opt_x:"+str(x)
                       + ", hmm_obj.startprob_:"
                       + str([str(val) for val in hmm_obj.startprob_])
                       + ", hmm_obj.transmat_:"
                       + str([str(val) for val in hmm_obj.transmat_])
                       + ", hmm_obj.means_:"
                       + str([str(val) for val in hmm_obj.means_])
                       + ", hmm_obj.covars_:"
                       + str([str(val) for val in hmm_obj.covars_])
                       + " [llh/hmmlearn.score] command: "+str(cmd_arg))
            # score()がエラーとなった場合もinfを返す。
            return np.inf
        return tmp_score

    # 最適化
    debug_print("start opt.geneticoptimize(). command: "+str(cmd_arg))
    try:
        s = opt.geneticoptimize(domain, llh, popsize=popsize, maxiter=maxiter,
                                elite=0.3, mutprob=0.5, cmd_arg=cmd_arg)
    except:
        error_exit("function error trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [opt.geneticoptimize] command: "+str(cmd_arg))
    if s == -1:
        raise Exception
    debug_print("end opt.geneticoptimize(). command: "+str(cmd_arg))
    return G_HMM_conv_para(s, ncompo, cmd_arg=cmd_arg)


# 上記の最適化関数で使用する。最適化された配列をパラメータに変換する。
def G_HMM_conv_para(s, ncompo, cmd_arg="nan"):
    try:
        x_startprob = s[0:ncompo]
        x_transmat = s[ncompo:(ncompo+ncompo**2)]
        x_means = s[(ncompo+ncompo**2):(ncompo**2+ncompo*2)]
        x_covars = s[(ncompo**2+ncompo*2):(ncompo**2+3*ncompo)]

        ret_startprob = mk_startprob(x_startprob)
        ret_transmat = mk_transmat(x_transmat, ncompo)
        ret_means = np.array(x_means).reshape(ncompo, 1)/10.0
        ret_covars = np.array(x_covars).reshape(ncompo, 1, 1)/100.0
    except:
        error_exit("function error. trace:"
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [G_HMM_conv_para] command: "+str(cmd_arg))
    return ret_startprob, ret_transmat, ret_means, ret_covars


# 目的：HMMのベストの状態数を探索する
#       最大状態数に達するかAIC/BICが減少しなくなるまで状態数を増やしていく
# 引数：対象データ(配列)、最大状態数、HMMパラメータ推定関数、初期値の最適化有無
# 返り値1：ベストの状態数でfitしたモデルオブジェクト
# 返り値2：ベストの状態数のパラメ多推定値を1行のDataFrameにしたもの
# 返り値3：各状態数の（ベスト以外も含む）パラメータ推定値をDataFrameにしたもの
def HMM_search_best_state(in_data, maxstate, para_est_func, opt=False,
                          cmd_arg="nan"):
    state_num_list = range(2, int(maxstate+1))

    # 入れ物の初期化（状態数ループ前）
    hmm_obj_list = list(np.repeat(np.nan, maxstate+1))
    hmm_resDF = pd.DataFrame()

    # パラメータ推定
    best_score = np.Inf
    for tmp_state in state_num_list:
        debug_print("start parameter estimation of "+str(tmp_state)
                    + "-state-HMM. command: "+str(cmd_arg))
        hmm_obj_list[tmp_state], tmp_resDF \
            = para_est_func(in_data, state_num=tmp_state, opt=opt, outDF=True,
                            cmd_arg=cmd_arg)
        debug_print("end parameter estimation of "+str(tmp_state)
                    + "-state-HMM. command: "+str(cmd_arg))
        hmm_resDF = pd.concat([hmm_resDF, tmp_resDF])
        # AIC/BICが減少していなければ終了
        tmp_score = tmp_resDF["bic"][0]
        if best_score <= tmp_score:
            break
        else:
            best_score = tmp_score

    # 最小AIC/BIC値の行を抽出 #複数存在する場合は状態数が小さい1行を抽出
    hmm_resDFmin = hmm_resDF[hmm_resDF['bic'] == min(hmm_resDF['bic'])].head(1)
    hmm_resDFmin = hmm_resDFmin.dropna(axis=1)
    best_state = int(hmm_resDFmin["State_num"])  # AIC/BIC最小の状態数
    hmm_obj_best = hmm_obj_list[best_state]
    return hmm_obj_best, hmm_resDFmin, hmm_resDF


# HMM_search_best_stateのwrapper関数  GaussianHMM
def G_HMM_search_BS(in_data, maxstate, opt=False, cmd_arg="nan"):
    return HMM_search_best_state(in_data, maxstate, G_HMM_para_est, opt=opt,
                                 cmd_arg=cmd_arg)


# 目的：データのヒストグラムとhmmの各状態の確率分布を重ねて出力
# 引数：データ（配列）、モデルオブジェクト、出力ファイル名（フルパス）
# 返り値：なし
def plot_G_HMMdist(data, model, path, cmd_arg="nan"):
    # ヒストグラムプロット
    fig, ax1 = plt.subplots(figsize=(10, 5))
    tmp_bins = int((max(data)-min(data))/2)
    l_str = "histogram of data"
    # データの範囲に応じてbins幅を調整
    if tmp_bins >= 30:
        ax1.hist(data, bins=30, color="gray", alpha=0.5, label=l_str)
    elif tmp_bins >= 20:
        ax1.hist(data, bins=tmp_bins, color="gray", alpha=0.5, label=l_str)
    elif tmp_bins >= 10:
        ax1.hist(data, bins=20, color="gray", alpha=0.5, label=l_str)
    elif tmp_bins >= 1:
        ax1.hist(data, bins=15, color="gray", alpha=0.5, label=l_str)
    else:
        ax1.hist(data, color="gray", alpha=0.5, label=l_str)
    plt.xlabel("active power")
    # 確率分布プロット
    var = np.ravel(model.covars_)
    sigmas = np.sqrt(var)
    myus = np.ravel(model.means_)
    try:
        stationary_stateprob = getFSS(model.transmat_, cmd_arg=cmd_arg)
    except:
        stationary_stateprob = np.repeat(1.0/len(myus))
    ax2 = ax1.twinx()
    state_num = 0
    # x = np.arange(min(data), max(data), 0.001)
    x = np.arange(0, max(data), 0.001)
    for v in zip(sigmas, myus, stationary_stateprob):
        # 正規分布のプロット用データを作成。定常確率を掛けてサイズを調整
        y = (1./np.sqrt(2*np.pi*v[0])) * np.exp(-(x - v[1])**2/2/v[0]) * v[2]
        ax2.plot(x, y, label="HMM state "+str(state_num))
        plt.ylim(0, 0.5)
        ax2.legend(loc="best", prop={'size': 10})
        state_num = state_num+1
    plt.title("histogram of data / distribution of model")
    # プロット出力
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')  # グラフを閉じて負荷軽減
    plt.close(fig)  # グラフを閉じて負荷軽減


# 目的：データの系列と隠れ状態の系列を重ねて出力
# 引数：データ(配列)、隠れ状態の系列、開始点、終了点、出力ファイル名(フルパス)
# 返り値：なし
def plot_HMMfit(data, state_data, plot_start, plot_end, path):
    state_num = np.max(np.array(pd.DataFrame(state_data).dropna()))
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(data[plot_start:plot_end], color="blue")
    ax2 = ax1.twinx()
    ax2.plot(state_data[plot_start:plot_end], color="red")
    plt.ylim(-0.1, state_num+0.1)
    plt.yticks(range(0, int(state_num+1)))
    plt.title("data plot / HMM state plot")
    # プロット出力
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')  # グラフを閉じて負荷軽減
    plt.close(fig)  # グラフを閉じて負荷軽減


# 目的：初期状態確率を生成する。要素に0を含むとエラーになる。
# 引数：パラメータ(1-100)の配列(state数)
# 返り値：state数次元の初期状態確率
def mk_startprob(x):
    ret = []
    for i in np.arange(len(x)):
        ret.append(float(x[i])/sum(x))
    # ret=list(ret/sum(ret))
    return ret


# 目的：推移行列を生成する。要素に0を含むとエラーになる。
# 引数：パラメータ(1-100)の配列(S*S))
# 返り値：S*Sの推移確率行列
def mk_transmat(x, y):
    y = int(y)
    ret = []
    for i in range(0, y):
        # 対角成分の値をまず設定。他は0とする。
        tmp_list = list(np.repeat(0, y))
        tmp_list[i] = 4/5.0
        # mk_startprob()を用いて、残りの確率を各要素に割り振る
        tmp_ret = mk_startprob(x[(i*(y)):((i+1)*(y))])
        tmp_ret2 = []
        for j in range(0, y):
            tmp_ret2.append(tmp_ret[j]/5.0)
        tmp_ret = list(np.array(tmp_list)+np.array(tmp_ret2))
        if len(ret) == 0:
            ret = tmp_ret
        else:
            ret = np.vstack([ret, tmp_ret])
    return ret


# 自作関数 GMM用

# 目的：GMMHMMのパラメータ推定（平均、分散、初期確率、推移行列） 混合数は2で固定
# 引数：対象データ(配列)、HMMの状態数
# 返り値1：fitしたモデルオブジェクト
# 返り値2；パラメ多推定値を1行のDataFrameにしたもの
def GMM_HMM_para_est(in_data, state_num=2, opt=False, outDF=False, nmix=2,
                     cmd_arg="nan"):
    state_list = np.arange(state_num)
    # hmmlearn用データ加工
    if len(np.shape(in_data)) == 1:
        in_data = np.c_[in_data]
    elif len(np.shape(in_data)) == 2:
        if np.shape(in_data)[1] == 1:
            in_data = np.c_[in_data]
        else:
            error_exit("shape of in_data is unexpected. np.shape(in_data): "
                       + str(np.shape(in_data))+" [GMM_HMM_para_est]"
                       + " command: "+str(cmd_arg))
    else:
        error_exit("shape of input data is unexpected. np.shape(in_data): "
                   + str(np.shape(in_data))+" [GMM_HMM_para_est] command: "
                   + str(cmd_arg))
    if opt is True:  # 初期値最適化あり
        hmm_obj = hmm.GMMHMM(n_components=state_num, n_mix=2,
                             init_params="")
        debug_print("start GMM_HMM_optimization(). command: "+str(cmd_arg))
        hmm_obj.startprob_, hmm_obj.transmat_, tmp_means, tmp_covars, \
            tmp_weights = GMM_HMM_optimization(in_data, state_num,
                                               cmd_arg=cmd_arg)
        for i in state_list:
            hmm_obj.gmms_[i].means_ = tmp_means[i].reshape(nmix, 1)
            hmm_obj.gmms_[i].covars_ = tmp_covars[i].reshape(nmix, 1)
            hmm_obj.gmms_[i].weights_ = tmp_weights[i]

        state_list = range(len(hmm_obj.startprob_))
        tmp_means = [hmm_obj.gmms_[i].means_ for i in state_list]
        tmp_covars = [hmm_obj.gmms_[i].covars_ for i in state_list]
        tmp_weights = [hmm_obj.gmms_[i].weights_ for i in state_list]
        debug_print("hmm_obj.startprob_):"
                    + str([str(x) for x in hmm_obj.startprob_])
                    + ", hmm_obj.transmat_:"
                    + str([str(x) for x in hmm_obj.transmat_])
                    + ", hmm_obj.means_:"
                    + str([str(x) for x in tmp_means])
                    + ", hmm_obj.covars_:"
                    + str([str(x) for x in tmp_covars])
                    + ", hmm_obj.weights_:"
                    + str([str(x) for x in tmp_weights]))
        debug_print("end GMM_HMM_optimization(). command: "+str(cmd_arg))
    else:  # なし
        hmm_obj = hmm.GMMHMM(n_components=state_num, n_mix=nmix,
                             init_params="stcw")
        # 初期化
        init_m = \
            np.array([int(max(in_data)/x)
                      for x in range(1, int((state_num*nmix)+1))][::-1])
        for i in state_list:
            hmm_obj.gmms_[i].means_ = \
                init_m[(nmix*i):(nmix*(i+1))].reshape(nmix, 1)

    # モデリング
    try:
        hmm_obj.fit(in_data)
    except Exception as exception:
        error_exit("function error. msg: "+str(exception)
                   + ", hmm_obj.startprob_:"
                   + str([str(val) for val in hmm_obj.startprob_])
                   + ", hmm_obj.transmat_:"
                   + str([str(val) for val in hmm_obj.transmat_])
                   + ", hmm_obj.gmms_[0].weights_:"
                   + str(hmm_obj.gmms_[0].weights_)
                   + " [llh/hmmlearn.fit] command: "+str(cmd_arg))
    # ValueError: rows of transmat_ must sum to 1.0 の対策でrowsの合計を1にする
    hmm_obj.transmat_ = (hmm_obj.transmat_.T/sum(hmm_obj.transmat_.T)).T

    # meanの小さい順に各状態のパラメータ値を入れ替える
    indexer = \
        np.argsort([np.mean(hmm_obj.gmms_[i].means_) for i in state_list])
    hmm_obj.startprob_ = hmm_obj.startprob_[indexer]
    hmm_obj.transmat_ = hmm_obj.transmat_[indexer].T[indexer].T
    hmm_obj.gmms_ = [hmm_obj.gmms_[i] for i in indexer]

    if outDF is False:
        # 空のDFを返す
        hmm_resDF = pd.DataFrame()
    else:
        # 尤度関係
        # AIC=(-2)*LLH+2*param_num
        # パラメータ数(param_num)：
        #   分布のパラメータ数*状態数(K*s)+初期状態確率(s-1)+推移行列(s*(s-1))
        #   ⇒s^2 + ks - 1
        # k=各平均*状態数*mix数、各分散*状態数*mix数、mix割合*状態数
        LLH = hmm_obj.score(in_data)
        # AIC = (-2)*LLH+2*(state_num**2+state_num*2*nmix-1)
        param_num = state_num**2+state_num*2*nmix+state_num*nmix-1
        BIC = (-2)*LLH+(np.log(len(in_data)))*param_num
        # AIC_LLH = pd.DataFrame({"State_num": [state_num],
        #                       "aic": [AIC], "llh": [LLH]})
        BIC_LLH = pd.DataFrame({"State_num": [state_num],
                               "bic": [BIC], "llh": [LLH]})

        # パラメータ取得
        initProb = pd.DataFrame(np.ravel(hmm_obj.startprob_)).T
        initProb.columns = ["startprob_"+str(x) for x in state_list]
        transMat = pd.DataFrame(np.ravel(hmm_obj.transmat_)).T
        transMat.columns = ["transprob_"+str(x)+"_"+str(y)
                            for x in state_list for y in state_list]
        try:
            sProb = pd.DataFrame(np.ravel(getFSS(hmm_obj.transmat_,
                                                 cmd_arg=cmd_arg))).T
        except:
            sProb = pd.DataFrame(np.repeat(np.nan, state_num)).T
        sProb.columns = ["stationaryProb_"+str(x) for x in state_list]
        x_means = \
            pd.DataFrame(np.ravel([hmm_obj.gmms_[i].means_
                                   for i in state_list])).T
        x_means.columns = ["mean_state"+str(x)+"_"+str(y)
                           for x in state_list for y in np.arange(nmix)]
        x_vars = \
            pd.DataFrame(np.ravel([hmm_obj.gmms_[i].covars_
                                   for i in state_list])).T
        x_vars.columns = ["var_state"+str(x)+"_"+str(y)
                          for x in state_list for y in np.arange(nmix)]
        x_weights = \
            pd.DataFrame(np.ravel([hmm_obj.gmms_[i].weights_
                                   for i in state_list])).T
        x_weights.columns = ["weight_state"+str(x)+"_"+str(y)
                             for x in state_list for y in np.arange(nmix)]

        # 結合
        hmm_resDF = pd.concat([BIC_LLH, initProb, transMat, sProb, x_means,
                               x_vars, x_weights], axis=1)

    return hmm_obj, hmm_resDF


# 目的：GMM_HMMパラメータ推定で用いる初期値の最適化
# 引数：データ配列、HMMの状態数
# 返り値：各パラメータの初期値（初期状態確率、推移行列、平均、分散、mix割合）
def GMM_HMM_optimization(data, ncompo, popsize=GA_pop, maxiter=GA_gen, nmix=2,
                         cmd_arg="nan"):
    # hmmlearn用データ加工
    if len(np.shape(data)) == 1:
        in_data = np.c_[data]
    elif len(np.shape(data)) == 2:
        if np.shape(data)[1] == 1:
            in_data = np.c_[data]
        else:
            error_exit("shape of in_data is unexpected. np.shape(data): "
                       + str(np.shape(data))+" [GMM_HMM_optimization]"
                       + " command: "+str(cmd_arg))
    else:
        error_exit("shape of in_data is unexpected. np.shape(data): "
                   + str(np.shape(data))+" [GMM_HMM_optimization]"
                   + " command: "+str(cmd_arg))

    # 最適化の領域定義
    dom_s = np.ravel([(1, 100)]*(ncompo))
    dom_t = np.ravel([(1, 100)]*(ncompo**2))
    dom_m = np.ravel([(int(min(in_data)*10),
                       int(max(in_data)*10))]*ncompo*nmix)
    in_var = np.var(in_data)
    dom_c = np.ravel([(max(1, int(in_var/10)),
                       max(1, int(in_var*100)))]*ncompo*nmix)
    dom_w = np.ravel([(1, 100)]*(ncompo*nmix))

    # (パラメータ数,上限下限(=2))でreshapeする
    param_num = ncompo**2+ncompo+3*nmix*ncompo
    domain = \
        np.hstack([dom_s, dom_t, dom_m, dom_c, dom_w]).reshape(param_num, 2)

    # 目的関数定義 LLHを最大化する
    # opt.geneticoptimizeに渡す目的関数の引数は、最適化する配列の1つしか渡せない
    def llh(x):
        if isinstance(x, type(None)):
            warn_print("arg type is None [llh] command: "+str(cmd_arg))
            return np.inf
        try:
            hmm_obj = hmm.GMMHMM(n_components=ncompo, n_mix=nmix,
                                 init_params="")
        except:
            error_exit("function error. trace:"
                       + traceback.format_exc(sys.exc_info()[2]))
        hmm_obj.startprob_, hmm_obj.transmat_, tmp_means, tmp_covars, \
            tmp_weights = GMM_HMM_conv_para(x, ncompo, cmd_arg=cmd_arg)
        for i in range(ncompo):
            hmm_obj.gmms_[i].means_ = tmp_means[i].reshape(nmix, 1)
            hmm_obj.gmms_[i].covars_ = tmp_covars[i].reshape(nmix, 1)
            hmm_obj.gmms_[i].weights_ = tmp_weights[i]

        # fit関数実行にタイムアウト処理を追加するためのラッパー関数
        @timeout(FUNCTION_TIMEOUT)
        def wrapper_hmmfit(data):
            hmm_obj.fit(data)

        try:
            wrapper_hmmfit(in_data)
        except Exception as exception:
            state_list = range(ncompo)
            tmp_means = [hmm_obj.gmms_[i].means_ for i in state_list]
            tmp_covars = [hmm_obj.gmms_[i].covars_ for i in state_list]
            tmp_weights = [hmm_obj.gmms_[i].weights_ for i in state_list]
            warn_print("function error. msg: "+str(exception)
                       + " opt_x:"+str(x)
                       + ", hmm_obj.startprob_:"
                       + str([str(val) for val in hmm_obj.startprob_])
                       + ", hmm_obj.transmat_:"
                       + str([str(val) for val in hmm_obj.transmat_])
                       + ", hmm_obj.means_:"
                       + str([str(val) for val in tmp_means])
                       + ", hmm_obj.covars_:"
                       + str([str(val) for val in tmp_covars])
                       + ", hmm_obj.weights_:"
                       + str([str(val) for val in tmp_weights])
                       + " [llh/hmmlearn.fit] command: "+str(cmd_arg))
            # fit()がエラーとなった場合は、infを返す
            return np.inf

        # fit()の結果、transmatの行または列が全て0になる場合もinfを返す
        trmat_DF = pd.DataFrame(hmm_obj.transmat_)
        if (trmat_DF.T.sum().min() == 0) | (trmat_DF.sum().min() == 0):
            warn_print("hmm_obj.transmat_ must sum to 1.0 ("
                       + str(np.array(trmat_DF.T.sum()))
                       + ","+str(np.array(trmat_DF.sum()))
                       + ") [llh] command: "+str(cmd_arg))
            return np.inf

        hmm_obj.transmat_ = (hmm_obj.transmat_.T/sum(hmm_obj.transmat_.T)).T

        # score関数実行にタイムアウト処理を追加するためのラッパー関数
        @timeout(FUNCTION_TIMEOUT)
        def wrapper_hmmscore(data):
            return hmm_obj.score(data)

        try:
            tmp_score = wrapper_hmmscore(in_data)*(-1)
        except Exception as exception:
            state_list = range(ncompo)
            tmp_means = [hmm_obj.gmms_[i].means_ for i in state_list]
            tmp_covars = [hmm_obj.gmms_[i].covars_ for i in state_list]
            tmp_weights = [hmm_obj.gmms_[i].weights_ for i in state_list]
            warn_print("function error. msg: "+str(exception)
                       + " opt_x:"+str(x)
                       + ", hmm_obj.startprob_:"
                       + str([str(val) for val in hmm_obj.startprob_])
                       + ", hmm_obj.transmat_:"
                       + str([str(val) for val in hmm_obj.transmat_])
                       + ", hmm_obj.means_:"
                       + str([str(val) for val in tmp_means])
                       + ", hmm_obj.covars_:"
                       + str([str(val) for val in tmp_covars])
                       + ", hmm_obj.weights_:"
                       + str([str(val) for val in tmp_weights])
                       + " [llh/hmmlearn.score] command: "+str(cmd_arg))
            # score()がエラーとなった場合もinfを返す。
            return np.inf
        return tmp_score

    # 最適化
    debug_print("start opt.geneticoptimize(). command: "+str(cmd_arg))
    try:
        s = opt.geneticoptimize(domain, llh, popsize=popsize, maxiter=maxiter,
                                elite=0.3, mutprob=0.5, cmd_arg=cmd_arg)
    except:
        error_exit("function error trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [opt.geneticoptimize] command: "+str(cmd_arg))
    if s == -1:
        raise Exception
    debug_print("end opt.geneticoptimize(). command: "+str(cmd_arg))
    return GMM_HMM_conv_para(s, ncompo, cmd_arg=cmd_arg)


# 上記の最適化関数で使用する。最適化された配列をパラメータに変換する。
def GMM_HMM_conv_para(s, ncompo, nmix=2, cmd_arg="nan"):
    try:
        x_startprob = s[0:ncompo]
        x_transmat = s[ncompo:(ncompo+ncompo**2)]
        x_means = s[(ncompo+ncompo**2):(ncompo**2+ncompo+nmix*ncompo)]
        x_covars = s[(ncompo**2+ncompo+nmix*ncompo):\
                     (ncompo**2+ncompo+2*nmix*ncompo)]
        x_weights = s[(ncompo**2+ncompo+2*nmix*ncompo):\
                      ((ncompo**2+ncompo+3*nmix*ncompo))]

        ret_startprob = mk_startprob(x_startprob)
        ret_transmat = mk_transmat(x_transmat, ncompo)
        ret_means = np.array(x_means).reshape(ncompo, nmix, 1)/10.0
        ret_covars = np.array(x_covars).reshape(ncompo, nmix, 1)/100.0
        tmp_weights = np.array(x_weights).reshape(ncompo, nmix)
        ret_weights = np.array([mk_startprob(tmp_weights[i]) \
                                for i in range(ncompo)])
    except:
        error_exit("function error. trace:"
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [G_HMM_conv_para] command: "+str(cmd_arg))
    return ret_startprob, ret_transmat, ret_means, ret_covars, ret_weights


# HMM_search_best_stateのwrapper関数
def GMM_HMM_search_BS(in_data, maxstate, opt=False, cmd_arg="nan"):
    return HMM_search_best_state(in_data, maxstate, GMM_HMM_para_est, opt=opt,
                                 cmd_arg=cmd_arg)


# 目的：データのヒストグラムとhmmの各状態の確率分布を重ねて出力
# 引数：データ（配列）、モデルオブジェクト、出力ファイル名（フルパス）
# 返り値：なし
def plot_GMM_HMMdist(data, model, path, cmd_arg="nan"):
    nmix = model.n_mix
    # ヒストグラムプロット
    fig, ax1 = plt.subplots(figsize=(10, 5))
    tmp_bins = int((max(data)-min(data))/2)
    l_str = "histogram of data"
    # データの範囲に応じてbins幅を調整
    if tmp_bins >= 100:
        ax1.hist(data, bins=50, color="gray", alpha=0.5, label=l_str)
    elif tmp_bins >= 20:
        ax1.hist(data, bins=tmp_bins, color="gray", alpha=0.5, label=l_str)
    elif tmp_bins >= 10:
        ax1.hist(data, bins=tmp_bins*2, color="gray", alpha=0.5, label=l_str)
    elif tmp_bins >= 1:
        ax1.hist(data, bins=tmp_bins*4, color="gray", alpha=0.5, label=l_str)
    else:
        ax1.hist(data, color="gray", alpha=0.5, label=l_str)
    # 確率分布プロット
    state_num = model.n_components
    state_list = np.arange(state_num)
    var = np.array([np.ravel(model.gmms_[i].covars_) for i in state_list])
    sigmas = np.sqrt(var)
    myus = np.array([np.ravel(model.gmms_[i].means_) for i in state_list])
    mixweight = \
        np.array([np.ravel(model.gmms_[i].weights_) for i in state_list])
    try:
        stationary_stateprob = getFSS(model.transmat_, cmd_arg=cmd_arg)
    except:
        stationary_stateprob = np.repeat(1.0/state_num)
    ax2 = ax1.twinx()
    state_num = 0
    x = np.arange(min(data), max(data), 0.001)
    for v in zip(sigmas, myus, mixweight, stationary_stateprob):
        y = 0
        for n in range(0, int(nmix)):
            y = y+(1./np.sqrt(2*np.pi*v[0][n]))*\
                  np.exp(-(x-v[1][n])**2/2/v[0][n])*v[2][n]
        y = y*v[3]  # 定常確率を掛けてサイズを調整
        ax2.plot(x, y, label="HMM state "+str(state_num))
        ax2.legend(loc="best", prop={'size': 10})
        state_num = state_num+1
    plt.title("histogram of data / distribution of model")
    # プロット出力
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')  # グラフを閉じて負荷軽減
    plt.close(fig)  # グラフを閉じて負荷軽減


# 自作関数 Multi用

# MultinomialHMMに当てはめるためのデータ条件
#
# That is ``obs`` should be an array of non-negative integers from
#        range ``[min(obs), max(obs)]``, such that each integer from the range
#        occurs in ``obs`` at least once.
#
# min(data)～max(data)を0～len(data)にmapしてモデリングしてから戻す処理が必要
def replace_data(x):
    k = np.unique(x)
    v = range(0, len(np.unique(x)))
    replace_dict = {k[i]: v[i] for i in v}

    def replace(x):
        return replace_dict[x]

    return np.c_[pd.DataFrame(x)[0].apply(replace).values]


def restore_data(x):
    k = np.unique(x)
    v = range(0, len(np.unique(x)))
    restore_dict = {v[i]: k[i] for i in v}

    def restore(x):
        return restore_dict[x]

    return np.c_[pd.DataFrame(x)[0].apply(restore).values]


# 目的：MultinomialHMMのパラメータ推定（各値の出力確率、初期確率、推移行列）
# 引数：対象データ(配列)、HMMの状態数
# 返り値1：fitしたモデルオブジェクト
# 返り値2；パラメ多推定値を1行のDataFrameにしたもの
def M_HMM_para_est(in_data, state_num=2, opt=False, cmd_arg="nan"):
    state_list = np.arange(state_num)
    # hmmlearn用データ加工
    if len(np.shape(in_data)) == 1:
        in_data = np.c_[in_data]
    elif len(np.shape(in_data)) == 2:
        if np.shape(in_data)[1] == 1:
            in_data = np.c_[in_data]
        else:
            error_exit("shape of in_data is unexpected. np.shape(in_data): "
                       + str(np.shape(in_data))+" [M_HMM_para_est]"
                       + " command: "+str(cmd_arg))
    else:
        error_exit("shape of in_data is unexpected. np.shape(in_data): "
                   + str(np.shape(in_data))+" [M_HMM_para_est] command: "
                   + str(cmd_arg))
    hmm_obj = hmm.MultinomialHMM(n_components=state_num)

    # データを変換　min(data)～max(data)を0～len(data)にmap
    uniq_vals = np.unique(in_data)
    in_data = replace_data(in_data)

    # モデリング
    hmm_obj.fit(in_data)
    # ValueError: rows of transmat_ must sum to 1.0 の対策でrowsの合計を1にする
    hmm_obj.transmat_ = (hmm_obj.transmat_.T/sum(hmm_obj.transmat_.T)).T

    # AIC=(-2)*LLH+2*param_num
    # パラメータ数(param_num)：
    #     分布のパラメータ数*状態数(K*s)+初期状態確率(s-1)+推移行列(s*(s-1))
    #     ⇒s^2 + ks - 1
    LLH = hmm_obj.score(in_data)
    K = hmm_obj.n_features
    # AIC = (-2)*LLH+2*(state_num**2+K*state_num-1)
    BIC = (-2)*LLH+(np.log(len(in_data)))*(state_num**2+K*state_num-1)

    # 尤度関係
    # AIC_LLH = pd.DataFrame({"State_num": [state_num],
    #                        "aic": [AIC], "llh": [LLH]})
    BIC_LLH = pd.DataFrame({"State_num": [state_num],
                           "bic": [BIC], "llh": [LLH]})

    # 最頻値の小さい順に各状態のパラメータ値を入れ替える
    indexer = \
        np.argsort([(hmm_obj.emissionprob_[i]).argmax() for i in state_list])
    hmm_obj.startprob_ = hmm_obj.startprob_[indexer]
    hmm_obj.transmat_ = hmm_obj.transmat_[indexer].T[indexer].T
    hmm_obj.emissionprob_ = hmm_obj.emissionprob_[indexer]

    # パラメータ取得
    initProb = pd.DataFrame(np.ravel(hmm_obj.startprob_)).T
    initProb.columns = ["startprob_"+str(x) for x in state_list]
    transMat = pd.DataFrame(np.ravel(hmm_obj.transmat_)).T
    transMat.columns = ["transprob_"+str(x)+"_"+str(y)
                        for x in state_list for y in state_list]
    try:
        sProb = pd.DataFrame(np.ravel(getFSS(hmm_obj.transmat_,
                                             cmd_arg=cmd_arg))).T
    except:
        sProb = pd.DataFrame(np.repeat(np.nan, state_num)).T
    sProb.columns = ["stationaryProb_"+str(x) for x in state_list]
    outProb = pd.DataFrame(np.ravel(hmm_obj.emissionprob_)).T
    outProb.columns = ["outprob_"+str(x)+"_"+str(y)
                       for x in state_list for y in uniq_vals]
    # 結合
    hmm_resDF = pd.concat([BIC_LLH, initProb, transMat, sProb, outProb],
                          axis=1)
    return hmm_obj, hmm_resDF


# HMM_search_best_stateのwrapper関数a  MultinomialHMM
def M_HMM_search_BS(in_data, maxstate, opt=False, cmd_arg="nan"):
    return HMM_search_best_state(in_data, maxstate, M_HMM_para_est, opt=opt,
                                 cmd_arg=cmd_arg)


# 目的：データのヒストグラムとhmmの各状態の確率分布を重ねて出力
# 引数：データ（配列）、モデルオブジェクト、出力ファイル名（フルパス）
# 返り値：なし
def plot_M_HMMdist(data, model, path, cmd_arg="nan"):
    # ヒストグラムプロット
    fig, ax1 = plt.subplots(figsize=(10, 5))
    tmp_bins = int((max(data)-min(data))/2)
    l_str = "histogram of data"
    # データの範囲に応じてbins幅を調整
    if tmp_bins >= 20:
        ax1.hist(data, bins=tmp_bins, color="gray", alpha=0.5, label=l_str)
    elif tmp_bins >= 10:
        ax1.hist(data, bins=tmp_bins*2, color="gray", alpha=0.5, label=l_str)
    elif tmp_bins >= 1:
        ax1.hist(data, bins=tmp_bins*4, color="gray", alpha=0.5, label=l_str)
    else:
        ax1.hist(data, color="gray", alpha=0.5, label=l_str)
    # 確率分布プロット
    emissionprob = model.emissionprob_
    try:
        stationary_stateprob = getFSS(model.transmat_, cmd_arg=cmd_arg)
    except:
        state_num = np.shape(emissionprob)[0]
        stationary_stateprob = np.repeat(1.0/state_num, state_num)
    x = np.unique(data)
    ax2 = ax1.twinx()
    state_num = 0
    for v in zip(emissionprob, stationary_stateprob):
        y = v[0]*v[1]
        ax2.plot(x, y, label="HMM state "+str(state_num))
        ax2.legend(loc="best", prop={'size': 10})
        state_num = state_num+1
    plt.title("histogram of data / distribution of model")
    # プロット出力
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')  # グラフを閉じて負荷軽減
    plt.close(fig)  # グラフを閉じて負荷軽減
