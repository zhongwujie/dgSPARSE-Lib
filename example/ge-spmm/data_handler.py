import subprocess
import glob
import os, shutil
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

class BalanceHandler:
  def __init__(self):
    sparsity = 0.99
    self.mtx_folder = f"../benchmark/input/balance/sp{sparsity:.2f}"
    self.out_folder = f"../benchmark/output/balance/sp{sparsity:.2f}"

  def exec(self):
    mtx_files = glob.glob(os.path.join(self.mtx_folder, "*.mtx"))
    sorted_files = sorted(mtx_files)
    algs = ["ROWCACHING_NNZBALANCE", "ROWCACHING_ROWBALANCE", "CUSPARSE"]
    N = 128
    data = {}

    for alg in algs:
      for mtx_file in sorted_files:
        print(f"{alg}: {mtx_file}")
        std_name = mtx_file.split("/")[-1].rstrip(".mtx")
        ret = single_exec(mtx_file, alg, N)
        throughput_val = get_throughput(ret)
        if std_name not in data:
          data[std_name] = []
        data[std_name].append(throughput_val)
    
    df = pd.DataFrame(data, index=algs)
    df["mean"] = df.mean(axis=1)
    df = df.round(2)
    os.makedirs(self.out_folder, exist_ok=True)
    csv_path = os.path.join(self.out_folder, "throughput.csv")
    df.to_csv(csv_path)

  def draw(self):
    sparsity = 0.97
    input_path = f"../benchmark/output/balance/sp{sparsity:.2f}/throughput.csv"
    output_path = f"../benchmark/output/balance/sp{sparsity:.2f}/throughput.png"
    df = pd.read_csv(input_path, index_col=0)
    df.columns = df.columns.str.replace("std_", "")
    new_idx = ["NNZBALANCE", "ROWBALANCE", "CUSPARSE"]
    df.index = new_idx
    df = df.transpose()
    draw_csv_bar(df, output_path, ylabel="Performance (gflops)", xlabel="Coefficient of Variation",
      margin=(0.85, 0.125, 0.125, 0.9))


class ReduceHandler:
  def __init__(self):
    sparsity = 0.50
    self.mtx_folder = f"../benchmark/input/reduce/sp{sparsity:.2f}"
    self.out_folder = f"../benchmark/output/reduce/sp{sparsity:.2f}"

  def exec(self):
    mtx_files = glob.glob(os.path.join(self.mtx_folder, "*.mtx"))
    sorted_files = sorted(mtx_files)
    algs = ["SEQREDUCE_ROWBALANCE", "PARREDUCE_ROWBALANCE", "CUSPARSE"]
    N = 128
    data = {}

    for alg in algs:
      for mtx_file in sorted_files:
        print(f"{alg}: {mtx_file}")
        MN_name = mtx_file.split("/")[-1].rstrip(".mtx")
        ret = single_exec(mtx_file, alg, N)
        throughput_val = get_throughput(ret)
        if MN_name not in data:
          data[MN_name] = []
        data[MN_name].append(throughput_val)
    
    df = pd.DataFrame(data, index=algs)
    df["mean"] = df.mean(axis=1)
    df = df.round(2)
    if not os.path.exists(self.out_folder):
      os.makedirs(self.out_folder)
    csv_path = os.path.join(self.out_folder, "throughput.csv")
    df.to_csv(csv_path)

  def draw(self):
    sparsity = 0.50
    input_path = f"../benchmark/output/reduce/sp{sparsity:.2f}/throughput.csv"
    output_path = f"../benchmark/output/reduce/sp{sparsity:.2f}/throughput.png"
    df = pd.read_csv(input_path, index_col=0)
    new_idx = ["SEQREDUCE", "PARREDUCE", "CUSPARSE"]
    df.index = new_idx
    df = df.transpose()
    draw_csv_bar(df, output_path, ylabel="Performance (gflops)", xlabel="Workload", rot=30,
      margin=(0.85, 0.125, 0.125, 0.9))


class TransposeHandler:
  def __init__(self):
    self.mtx_file = f"../data/p2p-Gnutella31.mtx"
    self.out_folder = f"../benchmark/output/transpose/p2p-Gnutella31"

  def exec(self):
    algs = ["SEQREDUCE_ROWBALANCE_NON_TRANSPOSE", "SEQREDUCE_ROWBALANCE", "CUSPARSE"]
    N_step = 32
    N_range = [N_step + i * N_step for i in range(8)]
    data = {}

    for alg in algs:
      for N in N_range:
        print(f"{alg}: {N}")
        N_name = f"N_{N}"
        ret = single_exec(self.mtx_file, alg, N)
        throughput_val = get_throughput(ret)
        if N_name not in data:
          data[N_name] = []
        data[N_name].append(throughput_val)
    
    df = pd.DataFrame(data, index=algs)
    df["mean"] = df.mean(axis=1)
    df = df.round(2)
    os.makedirs(self.out_folder, exist_ok=True)
    csv_path = os.path.join(self.out_folder, "throughput.csv")
    df.to_csv(csv_path)

  def draw(self):
    sparsity = 0.50
    input_path = f"../benchmark/output/transpose/p2p-Gnutella31/throughput.csv"
    output_path = f"../benchmark/output/transpose/p2p-Gnutella31/throughput.png"
    df = pd.read_csv(input_path, index_col=0)
    new_idx = ["Column-Major", "Row-Major", "CUSPARSE"]
    df.index = new_idx
    df = df.transpose()
    draw_csv_bar(df, output_path, ylabel="Performance (gflops)", xlabel="Workload", rot=30,
      margin=(0.85, 0.125, 0.125, 0.9))


def get_throughput(input_str: str) -> float:
  throughput_pattern = re.compile(r'Throughput (\d+\.\d+) \(gflops\)')
  match = throughput_pattern.search(input_str)
  if match:
    throughput_val = float(match.group(1))
  else:
    raise ValueError("Throughput value not found in the input string.")
  return throughput_val

def single_exec(mtx_path: str, alg: str, N: int) -> str:
  cmd = f"./spmm.out {mtx_path} {N} {alg}"
  ret = subprocess.run(cmd.split(), capture_output=True, text=True)
  output_str = ret.stdout
  return output_str

# draw the bar chart from the csv file
# margin: (top, bottom, left, right)
def draw_csv_bar(df: pd.DataFrame, save_path: str, ylabel: str, xlabel="Workload Configs", ylim=None, 
    figsize=(6, 4), rot=0, margin=(0.9, 0.125, 0.125, 0.9), 
    bar_color=["deepskyblue", 'orange',"mediumseagreen"]):
  ax = df.plot(kind='bar', rot=rot, figsize=figsize, color=bar_color)
  # set the margin
  plt.subplots_adjust(top=margin[0], bottom=margin[1], left=margin[2], right=margin[3])
  # set the labels
  ax.set_ylabel(ylabel)
  ax.set_xlabel(xlabel)
  # show the grid on y axis
  ax.set_axisbelow(True)
  ax.yaxis.grid(True, color="lightgrey", linewidth=0.75)
  ax.xaxis.grid(False)
  # hide the axis
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  # set the legend
  ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
  # set the ylim
  if(ylim):
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
  plt.savefig(save_path)

def run():
  hd = TransposeHandler()
  hd.draw()

if __name__ == "__main__":
  run()