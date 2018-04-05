# -*- coding: utf-8 -*-
#
# discription: define genetic algorithm function

import random
import numpy as np
import os
# ---時間関連---
import datetime
from timeout_decorator import timeout
# ログ用
from logging import getLogger, StreamHandler, FileHandler, INFO, WARN
import traceback
import sys

# ログ処理
cmd = "optimization"
pid = os.getpid()
logfile = "/tmp/hmm_tools_"+str(pid)+".log"
logger = getLogger(cmd)
Fhandler = FileHandler(logfile)
Fhandler.setLevel(INFO)
logger.addHandler(Fhandler)
Shandler = StreamHandler()
Shandler.setLevel(WARN)
logger.addHandler(Shandler)
logger.setLevel(INFO)


# 定数
FUNCTION_TIMEOUT = 3600
MULTITASK_TIMEOUT = 36000


# ログ出力関数
def error_print(msg):
    d = datetime.datetime.today()
    logger.error(d.strftime("%Y-%m-%d %H:%M:%S")+" ERROR "+cmd+" - "+str(msg))
    # 例外発生
    # raise Exception

def warn_print(msg):
  d = datetime.datetime.today()
  logger.warn(d.strftime("%Y-%m-%d %H:%M:%S")+" WARN "+cmd+" - "+str(msg))


def debug_print(msg):
  d = datetime.datetime.today()
  logger.info(d.strftime("%Y-%m-%d %H:%M:%S")+" INFO "+cmd+" - "+str(msg))


# 目的：遺伝的アルゴリズム(GA)による最適化を行う
# 引数：探索の領域(※1)、コスト関数(※2)、GAのパラメータ
# 返り値：最適解のリスト(※3)
#   ※1：領域は、変数の数分の範囲リストをもつリスト
#      例：[ [1,100], [1,100], [1,50], ... ]
#          →1つ目の変数の範囲は1～100
#   ※2：コスト関数は、整数のリストを引数に受け取り、スコアを返す関数を定義して
#        引数に渡す。
#   ※3：本関数では、まず、探索の領域の範囲内で整数のリストを生成し、コスト関数
#        に渡す。そして、コスト関数が返すスコアが最小になる整数のリストを、最適
#        解として返す。
def geneticoptimize(domain,costf,popsize=50,step=1,
                    mutprob=0.2,elite=0.2,maxiter=100,cmd_arg="nan"):

  random.seed(0)
  # costf関数にタイムアウト処理、エラー処理を追加
  @timeout(FUNCTION_TIMEOUT)
  def timed_costf(x):
    try:
      ret = costf(x)
    except:
      error_print("function error. trace: "
                  + traceback.format_exc(sys.exc_info()[2])
                  + " [costf] command: "+str(cmd_arg))
      raise Exception
    return ret

  # costf関数がタイムアウト/異常終了した場合にinfを返すためのラッパー関数
  def wrapper_costf(x):
    try:
      ret_score = timed_costf(x)
    except:
      warn_print("function error. trace: "
                 + traceback.format_exc(sys.exc_info()[2])
                 + " [timed_costf] command: "+str(cmd_arg))
      ret_score = np.inf
    return ret_score

  # costfのリスト内包による展処理にタイムアウト処理を追加するための関数
  @timeout(MULTITASK_TIMEOUT, use_signals=False)
  def listed_costf(pop):
    ret_scores=[(wrapper_costf(v),v) for v in pop]
    return ret_scores

  # Mutation Operation
  def mutate(vec):
    # 突然変異の個所をランダムに選定
    i=random.randint(0, len(domain)-1)
    # 1/2の確率で、該当箇所からstep値を引く。該当の値が下限値の場合は何もしない
    if random.random()<0.5 and vec[i]>domain[i][0]:
      return vec[0:i]+[vec[i]-step]+vec[i+1:]
    # 1/2の確率で、該当箇所にstep値を足す。該当箇所が上限値の場合は何もしない
    elif vec[i]<domain[i][1]:
      return vec[0:i]+[vec[i]+step]+vec[i+1:]
    # 下限値で引き算できなかった、上限値で足し算できなかった場合は、そのまま
    else:
      return vec
  
  # Crossover Operation
  def crossover(r1,r2):
    i=random.randint(1,len(domain)-2)
    return r1[0:i]+r2[i:]

  # Build the initial population
  pop=[]
  for i in range(popsize):
    vec=[random.randint(domain[i][0],domain[i][1]) 
         for i in range(len(domain))]
    pop.append(vec)
  
  # How many winners from each generation?
  topelite=int(elite*popsize)
  
  # Main loop
  for i in range(maxiter):
    debug_print("optimization roop "+str(i+1)+" start. command: "+str(cmd_arg))
    try:
      scores=listed_costf(pop)
    except Exception as exception:
      warn_print("function error. msg: "+str(exception)+" [listed_costf]"
                 + " command: "+str(cmd_arg))
      # scores=[(np.inf,v) for v in pop]
      # TIMEOUTしたらroop終了。もし以前のroopで結果が出ていればそれを返す
      break
    debug_print("optimization roop "+str(i+1)+" listed_costf() end. command: "
                + str(cmd_arg))
    scores.sort()
    ranked=[v for (s,v) in scores]
    
    # Start with the pure winners
    pop=ranked[0:topelite]
    
    # Add mutated and bred forms of the winners
    while len(pop)<popsize:
      if random.random()<mutprob:

        # Mutation
        c=random.randint(0,topelite)
        pop.append(mutate(ranked[c]))
      else:
      
        # Crossover
        c1=random.randint(0,topelite)
        c2=random.randint(0,topelite)
        pop.append(crossover(ranked[c1],ranked[c2]))
    
    # Print current best score
    # print scores[0][0]

    # ベストスコアが更新されなければ終了
    if "best_score" in locals():
      if best_score == scores[0][0]:
        debug_print("break optimization. command: "+str(cmd_arg))
        break
      else:
        best_score = scores[0][0]
    else:
      best_score = scores[0][0]
    
  if "scores" in locals():
    return scores[0][1]
  else:
    error_print("function error. msg: \"scores\" could not be calculated."
                +" [geneticoptimize] command: "+str(cmd_arg))
    return -1
