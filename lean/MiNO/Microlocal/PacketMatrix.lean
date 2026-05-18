import Mathlib
import MiNO.Microlocal.FinitePacket

open scoped BigOperators

namespace MiNO
namespace Microlocal

abbrev PacketMatrix (n : Nat) := Fin n → Fin n → ℝ

def matrixApply {n : Nat} (M : PacketMatrix n) (input : Coeff n) : Coeff n :=
  fun i => ∑ j, M i j * input j

def offTubeSet {n : Nat} (shell : Fin n → Fin n → Nat) (K : Nat) (i : Fin n) :
    Finset (Fin n) :=
  Finset.univ.filter fun j => K ≤ shell i j

def truncateByTube {n : Nat} (M : PacketMatrix n) (shell : Fin n → Fin n → Nat)
    (K : Nat) : PacketMatrix n :=
  fun i j => if shell i j < K then M i j else 0

def matrixTailApply {n : Nat} (M : PacketMatrix n) (shell : Fin n → Fin n → Nat)
    (K : Nat) (input : Coeff n) : Coeff n :=
  fun i => Finset.sum (offTubeSet shell K i) (fun j => M i j * input j)

def rowTailMass {n : Nat} (M : PacketMatrix n) (shell : Fin n → Fin n → Nat)
    (K : Nat) (i : Fin n) : ℝ :=
  Finset.sum (offTubeSet shell K i) (fun j => |M i j|)

theorem matrix_tail_l1_of_pointwise_bound
    {n : Nat}
    (M : PacketMatrix n)
    (shell : Fin n → Fin n → Nat)
    (K : Nat)
    (input : Coeff n)
    (tailBound : ℝ)
    (hpoint : ∀ i, |matrixTailApply M shell K input i| ≤ tailBound) :
    l1Norm (matrixTailApply M shell K input) ≤ n * tailBound := by
  calc
    l1Norm (matrixTailApply M shell K input)
      = ∑ i, |matrixTailApply M shell K input i| := by
          rfl
    _ ≤ ∑ _i : Fin n, tailBound := by
          refine Finset.sum_le_sum ?_
          intro i _hi
          exact hpoint i
    _ = n * tailBound := by
          simp

end Microlocal
end MiNO
