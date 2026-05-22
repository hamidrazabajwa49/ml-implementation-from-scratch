import math
import sys
import os
import random
import cmath

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  
from Vectors.vector import Vector

class Matrix:
    def __init__(self, data: list[list], tol: float = 1e-10):
        self.rows = [Vector(row) for row in data]
        self.tol = tol

        self.n_rows = len(self.rows)
        if self.n_rows == 0:
            self.n_cols = 0
        else:
            self.n_cols = len(self.rows[0])

        for row in self.rows:
            if len(row) != self.n_cols:
                raise ValueError("All rows must have same length")

    @property
    def shape(self):
        return (self.n_rows,self.n_cols)

    def __repr__(self):
        if self.n_rows == 0:
            return "Matrix([])"
        row_strings = [str(row.components) for row in self.rows]
        rows = ",\n    ".join(row_strings)
        return f"Matrix([\n    {rows}\n])"

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            return False
        tol = min(self.tol, other.tol)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if abs(self.rows[i].components[j] - other.rows[i].components[j]) > tol:
                    return False
        return True

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        result = [x + y for x, y in zip(self, other)]
        return Matrix([v.components for v in result])

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        result = [x - y for x, y in zip(self, other)]
        return Matrix([v.components for v in result])

    def __rsub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = [other * row for row in self.rows]
            return Matrix([row.components for row in result])

        if isinstance(other, Matrix):
            if self.n_cols != other.n_rows:
                raise ValueError(f"Dimension mismatch: {self.n_cols} != {other.n_rows}")
            result_rows = []
            for i in range(self.n_rows):
                current_row = []
                row_i = self.rows[i]
                for j in range(other.n_cols):
                    col_j = Vector([other.rows[k].components[j] for k in range(other.n_rows)])
                    current_row.append(row_i.dot(col_j))
                result_rows.append(current_row)
            return Matrix(result_rows)

        return NotImplemented

    __rmul__ = __mul__

    def transpose(self):
        if self.n_rows == 0:
            return Matrix([])
        new_rows = [
            Vector([self.rows[i].components[j] for i in range(self.n_rows)])
            for j in range(self.n_cols)
        ]
        return Matrix([row.components for row in new_rows])

    def copy(self):
        return Matrix([row.components.copy() for row in self.rows])

    @classmethod
    def zeros(cls, n_rows, n_cols):
        return cls([[0.0] * n_cols for _ in range(n_rows)])

    @classmethod
    def identity(cls, n):
        return cls([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])

    def row_echelon_form(self):
        mat = self.copy()
        pivot_row = 0
        n_rows = mat.n_rows
        n_cols = mat.n_cols

        for c in range(n_cols):
            pivot_index = -1
            for r in range(pivot_row, n_rows):
                if abs(mat.rows[r].components[c]) > self.tol:
                    pivot_index = r
                    break
            if pivot_index == -1:
                continue
            if pivot_index != pivot_row:
                mat.rows[pivot_row], mat.rows[pivot_index] = mat.rows[pivot_index], mat.rows[pivot_row]

            pivot_val = mat.rows[pivot_row].components[c]
            for i in range(pivot_row + 1, n_rows):
                if abs(mat.rows[i].components[c]) < self.tol:
                    continue
                factor = mat.rows[i].components[c] / pivot_val
                mat.rows[i] = mat.rows[i] - mat.rows[pivot_row] * factor

            pivot_row += 1
            if pivot_row == n_rows:
                break

        return mat

    def determinant(self):
        if self.n_rows != self.n_cols:
            raise ValueError("Determinant only defined for square matrices")

        mat = self.copy()
        n = mat.n_rows
        swap_count = 0

        for c in range(n):
            pivot_row = -1
            for r in range(c, n):
                if abs(mat.rows[r].components[c]) > self.tol:
                    pivot_row = r
                    break
            if pivot_row == -1:
                return 0.0
            if pivot_row != c:
                mat.rows[c], mat.rows[pivot_row] = mat.rows[pivot_row], mat.rows[c]
                swap_count += 1

            pivot_val = mat.rows[c].components[c]
            for i in range(c + 1, n):
                factor = mat.rows[i].components[c] / pivot_val
                mat.rows[i] = mat.rows[i] - mat.rows[c] * factor

        det = (-1) ** swap_count
        for i in range(n):
            det *= mat.rows[i].components[i]
        return det

    def inverse(self):
        if abs(self.determinant()) < self.tol:
            raise ValueError("Matrix is singular, no inverse exists.")

        n = self.n_rows
        I = Matrix.identity(n)
        aug_rows = [self.rows[i].components + I.rows[i].components for i in range(n)]
        aug = Matrix(aug_rows)

        for c in range(n):
            pivot_row = -1
            for r in range(c, n):
                if abs(aug.rows[r].components[c]) > self.tol:
                    pivot_row = r
                    break
            if pivot_row == -1:
                raise ValueError("Matrix is singular, no inverse exists.")
            if pivot_row != c:
                aug.rows[c], aug.rows[pivot_row] = aug.rows[pivot_row], aug.rows[c]

            pivot_val = aug.rows[c].components[c]
            aug.rows[c] = aug.rows[c] * (1.0 / pivot_val)

            for i in range(n):
                if i != c:
                    factor = aug.rows[i].components[c]
                    aug.rows[i] = aug.rows[i] - aug.rows[c] * factor

        return Matrix([row.components[n:] for row in aug.rows])

    def matvec(self, vec):
        if not isinstance(vec, Vector):
            raise ValueError("Input must be a Vector instance.")
        if self.n_cols != len(vec):
            raise ValueError(f"Dimension mismatch: matrix cols {self.n_cols} vs vector len {len(vec)}")
        return Vector([self.rows[i].dot(vec) for i in range(self.n_rows)])

    def power_iteration(self, max_iter=1000, tol=1e-8):
        if self.n_rows != self.n_cols:
            raise ValueError("Power iteration requires a square matrix.")

        n = self.n_rows
        if all(all(abs(x) < self.tol for x in row.components) for row in self.rows):
            raise ValueError("Power iteration is undefined for the zero matrix.")

        vec = Vector([0.0] * n)
        while vec.norm() < self.tol:
            vec = Vector([random.random() for _ in range(n)])
        v = vec.normalize()

        lambda_old = 0.0
        lambda_new = 0.0

        for _ in range(max_iter):
            v_new = self.matvec(v)
            lambda_new = v_new.dot(v)
            if v_new.norm() < self.tol:
                raise ValueError("Eigenvector vanished. Matrix may have zero eigenvalue.")
            v = v_new.normalize()
            if abs(lambda_new - lambda_old) < tol:
                return lambda_new, v
            lambda_old = lambda_new

        print(f"Warning: did not converge within {max_iter} iterations. Gap: {abs(lambda_new - lambda_old):.2e}")
        return lambda_new, v

    def trace(self):
        if self.n_rows != self.n_cols:
            raise ValueError("Trace requires a square matrix.")
        return sum(self.rows[i].components[i] for i in range(self.n_rows))

    def characteristic_poly(self):
        if self.n_rows != self.n_cols:
            raise ValueError("Matrix must be square.")

        n = self.n_rows

        if n == 2:
            return [1.0, -self.trace(), self.determinant()]

        if n == 3:
            def _minor_matrix(col):
                minor_data = []
                for r in range(3):
                    if r == col:
                        continue
                    row = [self.rows[r].components[j] for j in range(3) if j != col]
                    minor_data.append(row)
                return Matrix(minor_data)

            trace = self.trace()
            det = self.determinant()
            minor_sum = sum(_minor_matrix(k).determinant() for k in range(3))
            return [1.0, -trace, minor_sum, -det]

        raise NotImplementedError("Characteristic polynomial implemented only for 2x2 and 3x3.")

    def eigenvalues(self):
        if self.n_rows != self.n_cols:
            raise ValueError("Eigenvalues only defined for square matrices.")
        return self.qr_algorithm()

    def eigenvectors(self, lam):
        if self.n_rows != self.n_cols:
            raise ValueError("Input must be a square matrix.")

        n = self.n_rows
        I = Matrix.identity(n)
        M = self - (I * lam)
        R = M.row_echelon_form()

        def pivot_cols(mat):
            pivots = []
            for i in range(mat.n_rows):
                for j in range(mat.n_cols):
                    if abs(mat.rows[i].components[j]) > self.tol:
                        pivots.append(j)
                        break
            return pivots

        pivots = pivot_cols(R)
        free = [c for c in range(n) if c not in pivots]

        if not free:
            return []

        eigenvectors_list = []
        for free_var in free:
            x = [0.0] * n
            x[free_var] = 1.0

            for i in range(R.n_rows - 1, -1, -1):
                pivot = None
                for j in range(R.n_cols):
                    if abs(R.rows[i].components[j]) > self.tol:
                        pivot = j
                        break
                if pivot is None:
                    continue
                s = sum(R.rows[i].components[j] * x[j] for j in range(pivot + 1, R.n_cols))
                x[pivot] = -s / R.rows[i].components[pivot]

            eigenvectors_list.append(Vector(x))

        return eigenvectors_list

    def is_symmetric(self):
        return self == self.transpose()

    def diagonalize(self):
        if self.n_rows != self.n_cols:
            raise ValueError("Matrix must be square.")
        
        eigvals = self.eigenvalues()
        n = len(eigvals)
        
        if any(abs(lam.imag) > self.tol for lam in eigvals if isinstance(lam, complex)):
            raise NotImplementedError("Complex diagonalization not yet supported.")
        
        eigvecs = [self.eigenvectors(lam)[0] for lam in eigvals]
        row_matrix = Matrix([v.components for v in eigvecs])
        if abs(row_matrix.determinant()) < self.tol:
            raise ValueError("Matrix is not diagonalizable.")
        
        P = row_matrix.transpose()
        D = Matrix.zeros(n, n)
        for i, lam in enumerate(eigvals):
            D.rows[i].components[i] = float(lam.real if isinstance(lam, complex) else lam)       
        return P, D

    def spectral_theorem(self):
        result = {
            "symmetric": False,
            "real_eigenvalues": False,
            "orthogonal_eigenvectors": False,
        }

        if not self.is_symmetric():
            return result
        result["symmetric"] = True

        try:
            evals = self.eigenvalues()
        except (ValueError, NotImplementedError):
            return result

        real = all(
            isinstance(ev, (int, float)) or abs(ev.imag) < self.tol
            for ev in evals
        )
        result["real_eigenvalues"] = real

        try:
            evecs = [self.eigenvectors(ev)[0] for ev in evals]
        except ValueError:
            return result

        orthogonal = True
        n = len(evals)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(evals[i] - evals[j]) > self.tol:
                    if abs(evecs[i].dot(evecs[j])) > self.tol:
                        orthogonal = False
                        break
            if not orthogonal:
                break

        result["orthogonal_eigenvectors"] = orthogonal
        return result

    def columns(self):
        n_rows,n_cols=self.shape
        col=[]
        for j in range(n_cols):
            cols=[]
            for i in range(n_rows):
                num=self.rows[i].components[j]
                cols.append(num)
            cols=Vector(cols)
            col.append(cols)
        return (col)

    def qr_decompose(self):
        n = self.n_cols
        cols = self.columns()         
        q_cols = []
        R = Matrix.zeros(n, n)

        for i in range(n):
            a_i = cols[i]
            v = a_i

            for j in range(i):
                q_j = q_cols[j]
                r_ji = q_j.dot(a_i)
                R.rows[j].components[i] = r_ji
                v = v - q_j * r_ji

            r_ii = v.norm()
            if r_ii < 1e-12:

                q_i = Vector([0.0] * self.n_rows)
                R.rows[i].components[i] = 0.0
            else:
                R.rows[i].components[i] = r_ii
                q_i = v * (1.0 / r_ii)
            q_cols.append(q_i)

        Q = Matrix([q.components for q in q_cols]).transpose()
        return Q, R

    def qr_algorithm(self, max_iter=100, tol=1e-10):
        if self.n_rows != self.n_cols:
            raise ValueError("QR algorithm requires a square matrix.")

        M = self.copy()                     
        for _ in range(max_iter):
            Q, R = M.qr_decompose()
            M = R * Q                      

            off_diag = 0.0
            for i in range(M.n_rows):
                for j in range(i):
                    off_diag += M.rows[i].components[j] ** 2
            if off_diag < tol ** 2:
                break

        return [M.rows[i].components[i] for i in range(M.n_rows)]

    def diagonal(self):
        n_rows=self.n_rows
        diag_mat=Matrix.zeros(n_rows,n_rows)
        for i in range(n_rows):
            diag_mat.rows[i].components[i]=float(self.rows[i].components[i])
        return diag_mat

    def svd(self):
        m ,n = self.shape

        G = self.transpose() * self

        evals = G.eigenvalues()             
        evecs = []                          
        used_count = {}                     

        for lam in evals:
            vec_list = G.eigenvectors(lam)  
            if lam not in used_count:
                used_count[lam] = 0
            if used_count[lam] < len(vec_list):
                evecs.append(vec_list[used_count[lam]])
                used_count[lam] += 1
            else:
                evecs.append(Vector([0.0] * n))

        pairs = [(evals[i], i) for i in range(len(evals))]
        pairs.sort(reverse=True, key=lambda x: x[0])

        sorted_sigmas = []
        sorted_evecs = []
        for val, idx in pairs:
            s = max(0.0, val) ** 0.5       
            sorted_sigmas.append(s)
            sorted_evecs.append(evecs[idx])

        V = Matrix([v.components for v in sorted_evecs]).transpose()

        Sigma = Matrix.zeros(m, n)
        for i in range(min(m, n)):
            Sigma.rows[i].components[i] = sorted_sigmas[i]

        U_cols = []
        for i in range(min(m, n)):
            v_i = V.columns()[i]
            Av = self.matvec(v_i)
            norm_Av = Av.norm()

            if norm_Av > 1e-12:            
                u_i = Av * (1.0 / norm_Av)
            else:
                u_i = Vector([0.0] * m)    
            U_cols.append(u_i)

        U = Matrix([u.components for u in U_cols]).transpose()
        Vt = V.transpose()
        return U, Sigma, Vt

    @staticmethod
    def reconstruct(U,Sigma,Vt):
        return (U*Sigma*Vt)

    def low_rank_approx(self, k, U=None, Sigma=None, Vt=None):
        if U is None or Sigma is None or Vt is None:
            U, Sigma, Vt = self.svd()
        m, n = self.shape

        Sigma_k = Matrix.zeros(m, n)
        for i in range(min(m, n)):
            Sigma_k.rows[i].components[i] = Sigma.rows[i].components[i] if i < k else 0.0
        return U * Sigma_k * Vt

    def compression_ratio(self, k):
        n_rows, n_cols = self.shape
        if k < 0 or k > min(n_rows, n_cols):
            raise ValueError(f"k must be between 0 and {min(n_rows, n_cols)}")
        
        original = n_rows * n_cols
        compressed = n_rows * k + k + k * n_cols  
        ratio = compressed / original
        
        return {
            "ratio": ratio,
            "original_elements": original,
            "compressed_elements": compressed,
            "space_saved_percent": (1 - ratio) * 100 if ratio < 1 else 0
        }

    @classmethod
    def image_compression_demo(cls):
        pattern = [
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
        ]

        img = cls(pattern)
        m, n = img.shape
        max_k = min(m, n)

        print("=" * 60)
        print("SVD IMAGE COMPRESSION DEMO")
        print("=" * 60)
        print(f"\nOriginal image ({m}×{n}):")
        for row in img.rows:
            print("  " + " ".join(f"{int(x):3d}" for x in row.components))

        print("\nComputing SVD...")
        U, Sigma, Vt = img.svd()
        print("Done.\n")
        k_values = [1, 2, 3, 5, max_k]

        for k in k_values:
            print(f"{'─' * 40}")
            print(f"Rank-{k} Approximation")
            print(f"{'─' * 40}")

            approx = img.low_rank_approx(k, U=U, Sigma=Sigma, Vt=Vt)

            print("Reconstruction:")
            for row in approx.rows:
                vals = [max(0, min(255, int(round(x)))) for x in row.components]
                print("  " + " ".join(f"{v:3d}" for v in vals))

            stats = img.compression_ratio(k)
            print(f"Compression ratio: {stats['ratio']:.3f}")
            print(f"Space saved: {stats['space_saved_percent']:.1f}%")
            print(f"Storage: {stats['compressed_elements']} vs {stats['original_elements']} elements")

            diff = img - approx
            frob_error = sum(
                diff.rows[i].components[j] ** 2
                for i in range(m)
                for j in range(n)
            ) ** 0.5
            print(f"Reconstruction error (Frobenius): {frob_error:.2f}")

        print(f"\n{'=' * 60}")
        print(f"Key insight: With k={max_k}, the image is perfectly reconstructed (ratio=1.0)")
        print("Lower k values trade quality for storage savings.")
        print("=" * 60)
