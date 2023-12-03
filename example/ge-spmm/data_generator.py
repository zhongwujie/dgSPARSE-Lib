import numpy as np
import shutil
import os
import scipy.sparse as sps
import scipy.io as sio

class BalanceGenerator:
  def generate_mtx(self, M: int, N: int, nnz: int, nnz_std: float, mtx_path: str):
    # Create an array to hold the number of non-zero elements in each row
    nnz_per_row = np.random.normal(nnz/M, nnz_std, M).astype(int)
    
    # Ensure non-zero elements do not exceed the number of columns
    nnz_per_row = np.minimum(nnz_per_row, N)
    nnz_per_row = np.maximum(nnz_per_row, 0)
    
    # Adjust nnz_per_row so that its sum equals to nnz
    while np.sum(nnz_per_row) > nnz:
        i = np.random.randint(M)
        if nnz_per_row[i] > 0:
            nnz_per_row[i] -= 1
    while np.sum(nnz_per_row) < nnz:
        i = np.random.randint(M)
        if nnz_per_row[i] < N:
            nnz_per_row[i] += 1
            
    # Create the rows, columns and data arrays for the sparse matrix
    rows = np.repeat(np.arange(M), nnz_per_row)
    cols = np.random.randint(0, N, size=nnz)
    data = np.random.randint(1, 10, size=nnz)
    
    # Create the sparse matrix
    matrix = sps.coo_matrix((data, (rows, cols)), shape=(M, N))
    
    # Save the matrix to a .mtx file
    sio.mmwrite(mtx_path, matrix)

  def run(self):
    M = 1024
    N = 1024
    sparsity = 0.99
    density = 1 - sparsity
    nnz = int(M * N * density)
    cv_step = 0.1
    std_ratio = np.array([cv_step + i * cv_step for i in range(10)]) # coefficient of variation
    std_devs = N * density * std_ratio
    std_devs = list(std_devs)
    print(f"std_devs: {std_devs}")
    out_folder = f"../benchmark/input/balance/sp{sparsity:.2f}"
    if os.path.exists(out_folder):
      shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    for i in range(len(std_ratio)):
      std_dev = std_devs[i]
      cv = std_ratio[i]
      mtx_path = f"{out_folder}/std_{cv:.2f}.mtx"
      self.generate_mtx(M, N, nnz, std_dev, mtx_path)


class ReduceGenerator:
  def generate_mtx(self, M: int, N: int, nnz: int, mtx_path: str):    
    data = np.random.randint(1, 10, size=nnz)
    rows = np.random.randint(0, M, nnz)
    cols = np.random.randint(0, N, nnz)
    sparse_matrix = sps.coo_matrix((data, (rows, cols)), shape=(M, N))
    sio.mmwrite(mtx_path, sparse_matrix)

  def run(self):
    matrix_step = 256
    matrix_lens = [matrix_step + i * matrix_step for i in range(8)]
    sparsity = 0.50
    density = 1 - sparsity
    out_folder = f"../benchmark/input/reduce/sp{sparsity:.2f}"
    if os.path.exists(out_folder):
      shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    max_digit = len(str(max(matrix_lens)))
    for i in range(len(matrix_lens)):
      matrix_len = matrix_lens[i]
      digit = len(str(matrix_len))
      M = N = matrix_len
      nnz = int(M * N * density)
      mtx_path = f"{out_folder}/MN_" + "0" * (max_digit - digit) + f"{matrix_len}.mtx"
      self.generate_mtx(M, N, nnz, mtx_path)


def run():
  generator = ReduceGenerator()
  generator.run()

def test():
  matrix_step = 256
  matrix_lens = range(matrix_step, matrix_step * 8, matrix_step)
  print(matrix_lens)

if __name__ == '__main__':
  run()