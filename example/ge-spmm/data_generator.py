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
    sparsity = 0.20
    nnz = int(M * N * sparsity)
    std_ratio = np.arange(0.01, 0.11, 0.01) # coefficient of variation
    std_devs = N * sparsity * std_ratio
    std_devs = list(std_devs.astype(int))
    out_folder = f"../benchmark/input/balance/sp{sparsity:.2f}"
    if os.path.exists(out_folder):
      shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    for i in range(len(std_ratio)):
      std_dev = std_devs[i]
      cv = std_ratio[i]
      mtx_path = f"{out_folder}/std_{cv:.2f}.mtx"
      self.generate_mtx(M, N, nnz, std_dev, mtx_path)

def run():
  generator = BalanceGenerator()
  generator.run()

def test():
  a = np.array([0.1, 0.2])
  b = 2.2 * a
  print(b)

if __name__ == '__main__':
  run()