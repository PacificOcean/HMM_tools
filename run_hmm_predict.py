# -*- coding: utf-8 -*-
#
# discription: 作成済のHMMモデルから、隠れ状態を推定する
# input: CSV形式の時系列数値データ
# output:
#   - 隠れ状態の数列(csv)
#   - 元の数列と、隠れ状態の数列を重ねたプロット(png) ※output_png=Trueの場合のみ
# arguments:
#   argvs[1]: データファイルのパス
#   argvs[2]: モデリング対象のカラム名
#   argvs[3]: アウトプットの出力先ディレクトリのパス
#   argvs[4]: 作成済のHMMモデルのpickleファイル
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
import pickle
import datetime
import re  # ワイルドカード等の正規表現で使用

# モデリング関連
import hmm_utils

# ログ用
import traceback
from logging import getLogger, StreamHandler, FileHandler, INFO, WARN
cmd = "run_hmm_predict"
pid = str(os.getpid())
logfile = "/tmp/hmm_tools_"+pid+".log"
logger = getLogger(cmd)
Fhandler = FileHandler(logfile)
Fhandler.setLevel(INFO)
logger.addHandler(Fhandler)
Shandler = StreamHandler()
Shandler.setLevel(WARN)
logger.addHandler(Shandler)
logger.setLevel(INFO)

np.random.seed(0)

# 変数定義
# モデル
model = "G"   # Gaussian
# model = "GM"  # Gaussian-Mix
# model = "M"   # Multinomial  # 未サポート

# 固定パラメータ
output_png = False
div = 0.1  # データのスケール調整

if output_png is True:
    # 状態フィッティングプロット範囲
    plot_start = 0  # 描写開始位置
    plot_end = 96  # 描写終了位置


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

    def warn_print(msg):
        d = datetime.datetime.today()
        logger.warn(d.strftime("%Y-%m-%d %H:%M:%S")+" WARN "+cmd+" - "
                    + str(msg)+" command: "+arg_str)

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
        tgt_col = str(argvs[2])
        out_file = str(argvs[3])
        pickle_file = str(argvs[4])
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])+" [str]")

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
                   + " [os.path.dirname/exists/makedirs]")

    if output_png is True:
        # 状態分布プロット格納フォルダ
        save_hmmdist_dir = out_dir+"/distplot/"
        # 状態フィッティングプロット格納フォルダ
        save_hmmfitting_dir = out_dir+"/fitplot/"
        if not os.path.exists(save_hmmdist_dir):
            os.makedirs(save_hmmdist_dir)
        if not os.path.exists(save_hmmfitting_dir):
            os.makedirs(save_hmmfitting_dir)

    # main処理
    debug_print("start reading input file.")
    try:
        in_data = pd.read_csv(in_file)
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])+" [pd.read_csv]")
    debug_print("end reading input file.")

    if tgt_col not in in_data.columns:
        error_exit(1, tgt_col+" NOT in "+in_file+". [main]")

    # NAが含まれる場合の処理用
    NA_index_lst = in_data[in_data[tgt_col] != in_data[tgt_col]].index

    # データのスケール調整
    try:
        tmp_data = np.c_[np.array(in_data[tgt_col].dropna())]/div
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [np.c_/np.array/dropna]")
    if len(tmp_data) == 0:
        error_exit(1, tgt_col+" in "+in_file+" is empty. [main]")

    # モデルの読み込み
    debug_print("start loading pickle file.")
    try:
        with open(pickle_file, 'r') as f:
            hmm_obj = pickle.load(f)
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [open/pickle.load]")
    if model == "G":
        try:
            debug_print("hmm_obj.startprob_:"
                        + str([str(x) for x in hmm_obj.startprob_])
                        + ", hmm_obj.transmat_:"
                        + str([str(x) for x in hmm_obj.transmat_])
                        + ", hmm_obj.means_:"
                        + str([str(x) for x in hmm_obj.means_])
                        + ", hmm_obj.covars_:"
                        + str([str(x) for x in hmm_obj.covars_]))
        except:
            error_exit(2, "function error. trace: "
                       + traceback.format_exc(sys.exc_info()[2])
                       + " [hmm_obj.XXX]")
    debug_print("end loading pickle file.")

    # m-hmm時の変換。要修正：変換のルールを学習データと同じにする
    if model == "M":
        try:
            tmp_data = hmm_utils.replace_data(tmp_data)
        except:
            error_exit(2, "function error. trace: "
                       + traceback.format_exc(sys.exc_info()[2])
                       + " [hmm_utils.replace_data]")

    # 隠れ状態の推定
    debug_print("start hmm state estimation.")
    try:
        tmp_ans_state = hmm_obj.predict(tmp_data)
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [hmm_obj.predict]")
    debug_print("end hmm state estimation.")

    # NAが含まれる場合の処理
    try:
        if len(NA_index_lst) >= 1:
            tmp_ans_state_new = []
            cnt = 0
            for tmp_index in range(0, len(in_data)):
                if tmp_index in NA_index_lst:
                    tmp_ans_state_new.append(np.nan)
                else:
                    tmp_ans_state_new.append(tmp_ans_state[cnt])
                    cnt = cnt+1
            tmp_ans_state = tmp_ans_state_new
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [append/other proc]")

    # 状態推移列のDF化
    debug_print("start output hmmstate.")
    try:
        hmmstate_DF = pd.DataFrame(tmp_ans_state)
        hmmstate_DF.columns = [tgt_col+"_state"]
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [pd.DataFrame]")

    # 状態番号逆転の補正
    debug_print("start correction of reverse phenomenon.")
    try:
        # 元データと状態系列を結合
        merged_data = pd.concat([in_data, hmmstate_DF], axis=1)
        # 状態番号を取得
        state_lst = range(len(hmm_obj.startprob_))
        state_min = np.min(state_lst)
        state_max = np.max(state_lst)

        # 各状態の最大・最小値を取得
        max_vals = []
        min_vals = []
        for state in state_lst:
            tmp_merged = merged_data[merged_data[tgt_col+"_state"] == state]
            max_vals.append(tmp_merged[tgt_col].max())
            min_vals.append(tmp_merged[tgt_col].min())

        # リストから最小と最大を除く
        state_lst.remove(state_min)
        state_lst.remove(state_max)

        # 変数初期化
        reverse_before = pd.DataFrame()
        roop_cnt = 0
        while True:
            roop_cnt += 1

            # 逆転発生個所を探す。
            # 対象データの前後との差分項目を追加
            merged_data[tgt_col+"_diff"] = \
                merged_data[tgt_col] - merged_data[tgt_col].shift(1)
            merged_data[tgt_col+"_diff-1"] = \
                merged_data[tgt_col] - merged_data[tgt_col].shift(-1)

            # 対象データの状態番号の前後との差分項目を追加
            merged_data[tgt_col+"_s_diff"] = \
                merged_data[tgt_col+"_state"] \
                - merged_data[tgt_col+"_state"].shift(1)
            merged_data[tgt_col+"_s_diff-1"] = \
                merged_data[tgt_col+"_state"] \
                - merged_data[tgt_col+"_state"].shift(-1)

            # 前の時点に比べて、対象データの状態番号が下がっている、かつ、
            # 対象データが増えている時点のデータを抽出
            tmp_down_data = merged_data[merged_data[tgt_col+"_s_diff"] <= -1]
            down_data1 = tmp_down_data[tmp_down_data[tgt_col+"_diff"] > 0]

            # 前の時点に比べて、対象データの状態番号が上がっている、かつ、
            # 対象データが減っている時点のデータを抽出
            tmp_up_data = merged_data[merged_data[tgt_col+"_s_diff"] >= 1]
            up_data1 = tmp_up_data[tmp_up_data[tgt_col+"_diff"] < 0]

            # 次の時点と比べて、対象データの状態番号が下がっている、かつ、
            # 対象データが増えている時点のデータを抽出
            tmp_down_data = merged_data[merged_data[tgt_col+"_s_diff-1"] >= 1]
            down_data2 = tmp_down_data[tmp_down_data[tgt_col+"_diff-1"] < 0]

            # 次の時点に比べて、対象データの状態番号が上がっている、かつ、
            # 対象データが減っている時点のデータを抽出
            tmp_up_data = merged_data[merged_data[tgt_col+"_s_diff-1"] <= -1]
            up_data2 = tmp_up_data[tmp_up_data[tgt_col+"_diff-1"] > 0]

            # 逆転発生個所
            reverse = pd.concat([down_data2, down_data1, up_data2, up_data1])

            # 逆転が発生していない、または、これ以上の補正が出来ない場合は、ループ終了
            if (len(reverse) == 0):
                break
            elif (len(reverse) == len(reverse_before)):
                if (reverse.index == reverse_before.index).all():
                    break

            # 念のため、補正回数が、全レコード数に達した場合も、ループ終了
            if roop_cnt >= len(in_data):
                warn_print("Correction of reverse phenomenon did not end.")
                break

            # 逆転個所を保存
            reverse_before = reverse.copy()

            # 補正処理
            # 状態0(state_min)の最小値より、小さい値をとる状態があれば、状態0とする
            tmp_min = min_vals[state_min]
            if not np.isnan(tmp_min):  # NAの場合は補正不要
                tmp_reverse = reverse[reverse[tgt_col] < tmp_min]
                tmp_index = tmp_reverse.index
                if len(tmp_index) != 0:
                    merged_data.loc[tmp_index, tgt_col+"_state"] = state_min
                    continue

            # 状態2(state_min)の最大値より、大きい値をとる状態があれば、状態2とする
            tmp_max = max_vals[state_max]
            if not np.isnan(tmp_max):  # NAの場合は補正不要
                tmp_reverse = reverse[reverse[tgt_col] > tmp_max]
                tmp_index = tmp_reverse.index
                if len(tmp_index) != 0:
                    merged_data.loc[tmp_index, tgt_col+"_state"] = state_max
                    continue

            # 状態番号が最小でも最大でもない場合
            for state in state_lst:
                # 状態nの最小値よりも、小さい値をとる状態で、
                # かつ、状態番号がnよりも大きい場合は、状態をnとする
                tmp_min = min_vals[state]
                if not np.isnan(tmp_min):  # NAの場合は補正不要
                    tmp_reverse = reverse[(reverse[tgt_col] < tmp_min) &
                                          (reverse[tgt_col+"_state"] > state)]
                    tmp_index = tmp_reverse.index
                    if len(tmp_index) != 0:
                        merged_data.loc[tmp_index, tgt_col+"_state"] = state

                # 状態nの最大値よりも、大きい値をとる状態で、
                # かつ、状態番号がnよりも小さい場合は、状態をnとする
                tmp_max = max_vals[state]
                if not np.isnan(tmp_max):  # NAの場合は補正不要
                    tmp_reverse = reverse[(reverse[tgt_col] > tmp_max) &
                                          (reverse[tgt_col+"_state"] < state)]
                    tmp_index = tmp_reverse.index
                    if len(tmp_index) != 0:
                        merged_data.loc[tmp_index, tgt_col+"_state"] = state

        # 状態推移列の取得
        hmmstate_DF = pd.DataFrame(merged_data[tgt_col+"_state"])
        hmmstate_DF.columns = [tgt_col+"_state"]
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [main]")
    debug_print("end correction of reverse phenomenon.")

    # 状態系列のint化(NAが含まれていない場合のみ)
    try:
        if not hmmstate_DF.isnull().values.any():
            hmmstate_DF = hmmstate_DF.astype(int)
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [main]")

    # 状態推移列の出力
    try:
        hmmstate_DF.to_csv(out_file, index=False, header=True)
    except:
        error_exit(2, "function error. trace: "
                   + traceback.format_exc(sys.exc_info()[2])
                   + " [to_csv]")
    debug_print("end output hmmstate.")

    if output_png is True:
        debug_print("start output png file.")
        plot_data = np.c_[np.array(in_data[tgt_col])]

        # ファイル名の区切り記号
        spl_code = '\.csv'
        regex = re.compile(spl_code)

        # ファイル名取得
        tmp_fn = os.path.basename(out_file)
        file_name = regex.split(tmp_fn)[0]

        # 状態分布プロット
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
        # 状態フィッティングプロット
        tmp_output_fn = "hmmfit_"+file_name+".png"
        try:
            hmm_utils.plot_HMMfit(plot_data, tmp_ans_state, plot_start,
                                  min(plot_end, len(plot_data)),
                                  save_hmmfitting_dir+tmp_output_fn)
        except:
            error_exit(2, "function error. trace: "
                       + traceback.format_exc(sys.exc_info()[2])
                       + " [hmm_utils.plot_HMMfit]")
        debug_print("end output png file.")

    debug_print("end process.")

    os.remove(logfile)

    sys.exit(0)
