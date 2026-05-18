import Mathlib
import MiNO.Microlocal.FinitePacket

namespace MiNO
namespace Microlocal

def l1Stable
    {n m : Nat}
    (κ : ℝ)
    (map : Coeff n → Coeff m) : Prop :=
  ∀ a b, l1Norm (fun i => map a i - map b i) ≤ κ * l1Norm (fun j => a j - b j)

theorem l1_triangle_between
    {n : Nat}
    (a b c : Coeff n) :
    l1Norm (fun i => a i - c i)
      ≤ l1Norm (fun i => a i - b i) + l1Norm (fun i => b i - c i) := by
  have hpoint :
      ∀ i : Fin n, |a i - c i| ≤ |a i - b i| + |b i - c i| := by
    intro i
    calc
      |a i - c i| = |(a i - b i) + (b i - c i)| := by ring_nf
      _ ≤ |a i - b i| + |b i - c i| := by
        simpa [Real.norm_eq_abs] using norm_add_le (a i - b i) (b i - c i)
  calc
    l1Norm (fun i => a i - c i) = ∑ i, |a i - c i| := by rfl
    _ ≤ ∑ i, (|a i - b i| + |b i - c i|) := by
      refine Finset.sum_le_sum ?_
      intro i hi
      exact hpoint i
    _ = (∑ i, |a i - b i|) + ∑ i, |b i - c i| := by
      simp [Finset.sum_add_distrib]
    _ = l1Norm (fun i => a i - b i) + l1Norm (fun i => b i - c i) := by rfl

theorem cross_resolution_transfer_bound
    {nCoarse nFine : Nat}
    (restrict : Coeff nFine → Coeff nCoarse)
    (prolong : Coeff nCoarse → Coeff nFine)
    (referenceCoarse : Coeff nCoarse → Coeff nCoarse)
    (referenceFine : Coeff nFine → Coeff nFine)
    (approxCoarse : Coeff nCoarse → Coeff nCoarse)
    (inputFine : Coeff nFine)
    (κ coarseError transferError : ℝ)
    (hκ : 0 ≤ κ)
    (hstable : l1Stable κ prolong)
    (hcoarse :
      l1Norm (fun i => approxCoarse (restrict inputFine) i - referenceCoarse (restrict inputFine) i)
        ≤ coarseError)
    (htransfer :
      l1Norm (fun i => prolong (referenceCoarse (restrict inputFine)) i - referenceFine inputFine i)
        ≤ transferError) :
    l1Norm (fun i => prolong (approxCoarse (restrict inputFine)) i - referenceFine inputFine i)
      ≤ κ * coarseError + transferError := by
  have hprolong :
      l1Norm
          (fun i =>
            prolong (approxCoarse (restrict inputFine)) i
              - prolong (referenceCoarse (restrict inputFine)) i)
        ≤
          κ
            * l1Norm
                (fun i =>
                  approxCoarse (restrict inputFine) i - referenceCoarse (restrict inputFine) i) :=
    hstable (approxCoarse (restrict inputFine)) (referenceCoarse (restrict inputFine))
  have hscaled :
      l1Norm
          (fun i =>
            prolong (approxCoarse (restrict inputFine)) i
              - prolong (referenceCoarse (restrict inputFine)) i)
        ≤ κ * coarseError := by
    exact le_trans hprolong (mul_le_mul_of_nonneg_left hcoarse hκ)
  have htriangle :
      l1Norm (fun i => prolong (approxCoarse (restrict inputFine)) i - referenceFine inputFine i)
        ≤
          l1Norm
              (fun i =>
                prolong (approxCoarse (restrict inputFine)) i
                  - prolong (referenceCoarse (restrict inputFine)) i)
            +
            l1Norm (fun i => prolong (referenceCoarse (restrict inputFine)) i - referenceFine inputFine i) := by
    simpa using
      l1_triangle_between
        (a := prolong (approxCoarse (restrict inputFine)))
        (b := prolong (referenceCoarse (restrict inputFine)))
        (c := referenceFine inputFine)
  linarith

theorem cross_resolution_transfer_rate
    {nCoarse nFine : Nat}
    (restrict : Coeff nFine → Coeff nCoarse)
    (prolong : Coeff nCoarse → Coeff nFine)
    (referenceCoarse : Coeff nCoarse → Coeff nCoarse)
    (referenceFine : Coeff nFine → Coeff nFine)
    (approxCoarse : Coeff nCoarse → Coeff nCoarse)
    (inputFine : Coeff nFine)
    (κ coarseError transferError Ccoarse Ctransfer h : ℝ)
    (β α : Nat)
    (hκ : 0 ≤ κ)
    (hstable : l1Stable κ prolong)
    (hcoarse :
      l1Norm (fun i => approxCoarse (restrict inputFine) i - referenceCoarse (restrict inputFine) i)
        ≤ coarseError)
    (htransfer :
      l1Norm (fun i => prolong (referenceCoarse (restrict inputFine)) i - referenceFine inputFine i)
        ≤ transferError)
    (hcoarseRate : coarseError ≤ Ccoarse * h ^ β)
    (htransferRate : transferError ≤ Ctransfer * h ^ α) :
    l1Norm (fun i => prolong (approxCoarse (restrict inputFine)) i - referenceFine inputFine i)
      ≤ κ * (Ccoarse * h ^ β) + Ctransfer * h ^ α := by
  have hbase :=
    cross_resolution_transfer_bound
      (restrict := restrict)
      (prolong := prolong)
      (referenceCoarse := referenceCoarse)
      (referenceFine := referenceFine)
      (approxCoarse := approxCoarse)
      (inputFine := inputFine)
      (κ := κ)
      (coarseError := coarseError)
      (transferError := transferError)
      hκ hstable hcoarse htransfer
  have hscaled : κ * coarseError ≤ κ * (Ccoarse * h ^ β) := by
    exact mul_le_mul_of_nonneg_left hcoarseRate hκ
  linarith

end Microlocal
end MiNO
