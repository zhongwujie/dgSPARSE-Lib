import subprocess
import glob
import os
import re
from typing import List

class BalanceHandler:
  def __init__(self):
    sparsity = 0.20
    self.mtx_folder = f"../benchmark/input/balance/sp{sparsity:.2f}"

  def exec(self):
    mtx_files = glob.glob(os.path.join(self.mtx_folder, "*.mtx"))
    alg = "GESPMM_ALG_ROWCACHING_NNZBALANCE"
    N = 128
    for mtx_file in mtx_files:
      print(f"mtx_file: {mtx_file}")
      single_exec(mtx_file, alg, N)


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

def run():
  balance_hd = BalanceHandler()
  balance_hd.exec()

if __name__ == "__main__":
  run()