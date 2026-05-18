import Mathlib
import MiNO.Microlocal.FinitePacket

open scoped BigOperators

namespace MiNO
namespace Microlocal

/-- The identity canonical relation used by pseudodifferential branches. -/
def identityCanonical {n : Nat} : Fin n -> Fin n := fun i => i

theorem identityCanonical_apply {n : Nat} (i : Fin n) :
    identityCanonical i = i := by
  rfl

/-- A branchwise packet error after truncating FIO, pseudodifferential, and smoothing parts. -/
def unifiedBranchError {n : Nat}
    (fioError pdoError smoothingError : Coeff n) : Coeff n :=
  fun i => fioError i + pdoError i + smoothingError i

/--
Finite branch-composition bridge used by the restricted calculus theorem.
The continuous FIO and pseudodifferential estimates provide the pointwise
certificate `hpoint`; Lean checks that this certificate accumulates to the
advertised packet `ell^1` budget.
-/
theorem l1_unified_branch_error_bound
    {n : Nat}
    (fioError pdoError smoothingError : Coeff n)
    (branchBudget : Real)
    (hpoint : forall i,
      |unifiedBranchError fioError pdoError smoothingError i| <= branchBudget) :
    l1Norm (unifiedBranchError fioError pdoError smoothingError)
      <= n * branchBudget := by
  calc
    l1Norm (unifiedBranchError fioError pdoError smoothingError)
        = Finset.sum Finset.univ
          (fun i => |unifiedBranchError fioError pdoError smoothingError i|) := by
          rfl
    _ <= Finset.sum Finset.univ (fun _ : Fin n => branchBudget) := by
          exact Finset.sum_le_sum (by intro i _hi; exact hpoint i)
    _ = n * branchBudget := by
          simp

end Microlocal
end MiNO
