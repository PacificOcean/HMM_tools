# -*- coding: utf-8 -*-
#
# discription: HMMモデルを作成する。
# input: CSV形式の時系列数値データ
# output:
#   - HMMのモデルオブジェクト(pickle)
#   - HMMのパラメータリスト(csv) ※output_csv_png=Trueの場合のみ
#   - HMMのパラメータと分布のプロット(png) ※output_csv_png=Trueの場合のみ
# arguments:
#   argvs[1]: データファイルのパス
#   argvs[2]: モデリング対象のカラム名
#   argvs[3]: アウトプットの出力先のパス
#   argvs[4]: 隠れ状態数
# note:
#   データファイルの条件:
#    - pandasデータフレームとして読み込めること
#    - カラム名は必須、インデックス有無は問わない
#    - 1レコードが1時点を表す、時系列データを想定
#    - 対象カラムは、数値データであること
#    - 欠損があった場合は、欠損レコードを削除して動作する


# --基本モジュール--
import numpy as np
import pandas as pd
import os
import sys
import re  # ワイルドカード等の正規表現で使用
import pickle
import datetime

# モデリング関連
import hmm_utils

# ログ用
import traceback
from logging import getLogger, StreamHandler, FileHandler, INFO, ERROR
cmd = "run_hmm_modeling"
pid = str(os.getpid())
logfile = "/tmp/hmm_tools_"+pid+".log"
logger = getLogger(cmd)
Fhandler = FileHandler(logfile)
Fhandler.setLevel(INFO)
logger.addHandler(Fhandler)
Shandler = StreamHandler()
Shandler.setLevel(ERROR)
logger.addHandler(Shandler)
logger.setLevel(INFO)

np.random.seed(0)


# 変数定義
# モデル
model = "G"   # Gaussian
# model = "GM"  # Gaussian-Mix
# model = "M"   # Multinomial

# 固定パラメータ
output_csv_png = False  # パラメータリスト、分布プロットを出力するかどうか
do_state_opt = False  # 状態数の探索をするかどうか
div = 0.1  # データのスケール調整


# 処理開始
if __name__ == '__main__':
    # 引数取得
    argvs = sys.argv
    arg_str = ' '.join(map(str, argvs))

    # ログ関数定義
    def error_exit(code, msg):
        d = datetime.datetime.today()
        logger.error(d.strftime("%Y-%m-%d %H:%M:%S")+" ERROR "+cmd+" - "
                     + str(msg)+" command: "+arg_str)
        logfile2 = "/var/log/hmm_tools_"+d.strftime("%Y%m%d%H%M%S")+"_" \
                   + pid+".log"
        os.rename(logfile, logfile2)
        sys.exit(code)

    def debug_print(msg):
        d = datetime.datetime.today()
        logger.info(d.strftime("%Y-%m-%d %H:%M:%S")+" INFO "+cmd+" - "
                    + str(msg)+" command: "+arg_str)

    debug_print("start process.")
    # 引数チェック
    if len(argvs) <= 4:
        error_exit(1, "number of args is less than expected. [main]")

    try:
        in_file = str(argvs[1])
        tgt_colname = str(argvs[2])
        out_file = str(argvs[3])
        state_num = int(argvs[4])
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])+" [str/int]")

    # 状態数チェック
    if state_num <= 1:
        error_exit(1, "number of state is less than 2. state_num:"
                   + str(state_num)+". [main]")

    # パス関連
    try:
        out_dir = os.path.dirname(out_file)
        if len(out_dir) == 0:
            out_dir = "."
        elif not os.path.exists(out_dir):
            os.makedirs(out_dir)
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [os.path.dirname/exists/makedir]")

    if output_csv_png is True:
        save_hmmpara_dir = out_dir+"/hmmpara/"  # hmmpara格納フォルダ
        save_hmmdist_dir = out_dir+"/distplot/"  # 状態分布プロット格納フォルダ
        if not os.path.exists(save_hmmpara_dir):
            os.makedirs(save_hmmpara_dir)
        if not os.path.exists(save_hmmdist_dir):
            os.makedirs(save_hmmdist_dir)

    # main処理
    debug_print("start reading input file.")
    try:
        in_data = pd.read_csv(in_file, engine="python")
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])+" [pd.read_csv]")
    debug_print("end reading input file.")

    if tgt_colname not in in_data.columns:
        error_exit(1, tgt_colname+" NOT in "+in_file+". [main]")

    # データスケール調整
    try:
        tmp_data = np.c_[np.array(in_data[tgt_colname].dropna())]/div
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [np.c_/np.array/dropna]")

    if len(tmp_data) == 0:
        error_exit(1, tgt_colname+" in "+in_file+" is empty. [main]")

    if len(np.unique(tmp_data)) < state_num:
        error_exit(1, "Unique number of "+tgt_colname+" in "+in_file
                   + " is less then state number. [main]")

    debug_print("start HMM parameter estimation.")
    # パラメータ（状態数、出力確率分布のパラメータ、初期確率、推移行列）推定
    if model == "G":
        # 状態数探索なし版
        if do_state_opt is False:
            try:
                hmm_obj, hmm_resDF \
                    = hmm_utils.G_HMM_para_est(tmp_data, state_num=state_num,
                                               opt=True, outDF=output_csv_png,
                                               cmd_arg=arg_str)
            except:
                error_exit(2, "function error. trace: "
                           + traceback.format_exc(sys.exc_info()[2])
                           + " [hmm_utils.G_HMM_para_est]")
        # 状態数探索あり版
        else:
            max_state = state_num
            try:
                debug_print("start searching best state.")
                hmm_obj, hmm_resDFmin, hmm_resDF \
                    = hmm_utils.G_HMM_search_BS(tmp_data, max_state, opt=True,
                                                cmd_arg=arg_str)
                debug_print("end searching best state.")
            except:
                error_exit(2, "function error. trace: "
                           + traceback.format_exc(sys.exc_info()[2])
                           + " [hmm_utils.G_HMM_search_BS]")
        debug_print("hmm_obj.startprob_:"
                    + str([str(x) for x in hmm_obj.startprob_])
                    + ", hmm_obj.transmat_:"
                    + str([str(x) for x in hmm_obj.transmat_])
                    + ", hmm_obj.means_:"
                    + str([str(x) for x in hmm_obj.means_])
                    + ", hmm_obj.covars_:"
                    + str([str(x) for x in hmm_obj.covars_]))
    elif model == "GM":
        # 状態数探索なし版
        if do_state_opt is False:
            try:
                hmm_obj, hmm_resDF \
                    = hmm_utils.GMM_HMM_para_est(tmp_data, state_num=state_num,
                                                 opt=True, cmd_arg=arg_str)
            except:
                error_exit(2, "function error. trace: "
                           + traceback.format_exc(sys.exc_info()[2])
                           + " [hmm_utils.GMM_HMM_para_est]")
        # 状態数探索あり版
        else:
            max_state = state_num
            try:
                debug_print("start searching best state.")
                hmm_obj, hmm_resDFmin, hmm_resDF \
                    = hmm_utils.GMM_HMM_search_BS(tmp_data, max_state,
                                                  opt=True, cmd_arg=arg_str)
                debug_print("end searching best state.")
            except:
                error_exit(2, "function error. trace: "
                           + traceback.format_exc(sys.exc_info()[2])
                           + " [hmm_utils.G_HMM_search_BS]")
        state_list = range(len(hmm_obj.startprob_))
        tmp_mean = [hmm_obj.gmms_[i].means_ for i in state_list]
        tmp_covars = [hmm_obj.gmms_[i].covars_ for i in state_list]
        tmp_weights = [hmm_obj.gmms_[i].weights_ for i in state_list]
        debug_print("hmm_obj.startprob_):"
                    + str([str(x) for x in hmm_obj.startprob_])
                    + ", hmm_obj.transmat_:"
                    + str([str(x) for x in hmm_obj.transmat_])
                    + ", hmm_obj.means_:"
                    + str([str(x) for x in tmp_mean])
                    + ", hmm_obj.covars_:"
                    + str([str(x) for x in tmp_covars])
                    + ", hmm_obj.weights_:"
                    + str([str(x) for x in tmp_weights]))
    elif model == "M":
        try:
            hmm_obj, hmm_resDF \
                = hmm_utils.M_HMM_para_est(tmp_data, state_num=state_num,
                                           opt=False, cmd_arg=arg_str)
        except:
            error_exit(2, "function error. trace: "
                       + traceback.format_exc(sys.exc_info()[2])
                       + " [hmm_utils.M_HMM_para_est]")

    debug_print("end HMM parameter estimation.")

    debug_print("start dumping pickle file.")
    # オブジェクトファイルの出力
    try:
        with open(out_file, 'w') as f:
            pickle.dump(hmm_obj, f)
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [open/pickle.dump]")

    debug_print("end dumping pickle file.")

    if output_csv_png is True:
        debug_print("start output csv/png file.")
        # ファイル名の区切り記号
        spl_code3 = '\.'
        regex3 = re.compile(spl_code3)

        # ファイル名取得
        tmp_fn = os.path.basename(out_file)
        tmp_fn2 = regex3.split(tmp_fn)
        if len(tmp_fn2) <= 2:
            file_name = str(tmp_fn2[0])
        else:
            # 拡張子を取得して、それ以前を返す
            extension = tmp_fn2[len(tmp_fn2)-1]
            spl_code4 = '\.'+str(extension)
            regex4 = re.compile(spl_code4)
            tmp_fn3 = regex4.split(tmp_fn)
            file_name = str(tmp_fn3[0])

        # パラメータリストの出力
        tmp_output_fn = "hmmpara_"+file_name+".csv"
        hmm_resDF.to_csv(save_hmmpara_dir+tmp_output_fn, index=False)
        # 状態分布プロットの出力
        tmp_output_fn = "hmmdis_"+file_name+".png"

        if model == "G":
            try:
                hmm_utils.plot_G_HMMdist(tmp_data, hmm_obj,
                                         save_hmmdist_dir+tmp_output_fn,
                                         cmd_arg=arg_str)
            except:
                error_exit(2, "function error. trace: "
                           + traceback.format_exc(sys.exc_info()[2])
                           + " [hmm_utils.plot_G_HMMdist]")
        elif model == "GM":
            try:
                hmm_utils.plot_GMM_HMMdist(tmp_data, hmm_obj,
                                           save_hmmdist_dir+tmp_output_fn,
                                           cmd_arg=arg_str)
            except:
                error_exit(2, "function error. trace: "
                           + traceback.format_exc(sys.exc_info()[2])
                           + " [hmm_utils.plot_GMM_HMMdist]")

        elif model == "M":
            try:
                hmm_utils.plot_M_HMMdist(tmp_data, hmm_obj,
                                         save_hmmdist_dir+tmp_output_fn,
                                         cmd_arg=arg_str)
            except:
                error_exit(2, "function error. trace: "
                           + traceback.format_exc(sys.exc_info()[2])
                           + " [hmm_utils.plot_M_HMMdist]")

        debug_print("end output csv/png file.")

    debug_print("end process.")
