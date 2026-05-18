import Mathlib
import MiNO.Microlocal.FinitePacket

open scoped BigOperators

namespace MiNO
namespace Microlocal

structure StencilReferenceLayer (n K : Nat) where
  transport : Fin n → Fin K → Fin n
  symbol : Fin n → Fin K → ℝ

structure StencilApproxLayer (n K : Nat) where
  transport : Fin n → Fin K → Fin n
  symbol : Fin n → Fin K → ℝ
  residual : Fin n → ℝ

def applyStencilReference {n K : Nat} (layer : StencilReferenceLayer n K) (input : Coeff n) : Coeff n :=
  fun i => ∑ j, layer.symbol i j * input (layer.transport i j)

def applyStencilApprox {n K : Nat} (layer : StencilApproxLayer n K) (input : Coeff n) : Coeff n :=
  fun i => ∑ j, layer.symbol i j * input (layer.transport i j) + layer.residual i

theorem pointwise_stencil_propagation_bound
    {n K : Nat}
    (input : Coeff n)
    (reference : StencilReferenceLayer n K)
    (approx : StencilApproxLayer n K)
    (transportError symbolError truncationError coeffBound referenceSymbolBound : ℝ)
    (htransportNonneg : 0 ≤ transportError)
    (hsymbolNonneg : 0 ≤ symbolError)
    (hcoeffNonneg : 0 ≤ coeffBound)
    (hrefNonneg : 0 ≤ referenceSymbolBound)
    (htransport :
      ∀ i j, |input (approx.transport i j) - input (reference.transport i j)| ≤ transportError)
    (hsymbol : ∀ i j, |approx.symbol i j - reference.symbol i j| ≤ symbolError)
    (hresidual : ∀ i, |approx.residual i| ≤ truncationError)
    (hcoeff : ∀ i j, |input (approx.transport i j)| ≤ coeffBound)
    (href : ∀ i j, |reference.symbol i j| ≤ referenceSymbolBound) :
    ∀ i,
      |applyStencilApprox approx input i - applyStencilReference reference input i|
        ≤ (K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) + truncationError := by
  intro i
  let pairBound := referenceSymbolBound * transportError + symbolError * coeffBound
  have hpairNonneg : 0 ≤ pairBound := by
    nlinarith
  let mismatchTerm : Fin K → ℝ :=
    fun j =>
      reference.symbol i j * (input (approx.transport i j) - input (reference.transport i j))
        + (approx.symbol i j - reference.symbol i j) * input (approx.transport i j)
  have hsumrewrite :
      (∑ j, (approx.symbol i j * input (approx.transport i j) - reference.symbol i j * input (reference.transport i j)))
        =
      ∑ j, mismatchTerm j := by
    refine Finset.sum_congr rfl ?_
    intro j hj
    simp [mismatchTerm]
    ring
  have hdecomp :
      applyStencilApprox approx input i - applyStencilReference reference input i
        = (∑ j, mismatchTerm j) + approx.residual i := by
    calc
      applyStencilApprox approx input i - applyStencilReference reference input i
        = ((∑ j, approx.symbol i j * input (approx.transport i j)) -
            (∑ j, reference.symbol i j * input (reference.transport i j))) + approx.residual i := by
              simp [applyStencilApprox, applyStencilReference]
              ring
      _ = (∑ j, (approx.symbol i j * input (approx.transport i j) -
            reference.symbol i j * input (reference.transport i j))) + approx.residual i := by
              rw [Finset.sum_sub_distrib]
      _ = (∑ j, mismatchTerm j) + approx.residual i := by
              rw [hsumrewrite]
  rw [hdecomp]
  have hsumAbs :
      |∑ j, mismatchTerm j| ≤ ∑ j, |mismatchTerm j| := by
    simpa [Real.norm_eq_abs, mismatchTerm] using
      (norm_sum_le (s := Finset.univ) (f := mismatchTerm))
  have hterm :
      ∀ j : Fin K,
        |mismatchTerm j| ≤ pairBound := by
    intro j
    have h₁ :
        |reference.symbol i j * (input (approx.transport i j) - input (reference.transport i j))|
          ≤ referenceSymbolBound * transportError := by
      rw [abs_mul]
      exact mul_le_mul (href i j) (htransport i j) (abs_nonneg _) hrefNonneg
    have h₂ :
        |(approx.symbol i j - reference.symbol i j) * input (approx.transport i j)|
          ≤ symbolError * coeffBound := by
      rw [abs_mul]
      exact mul_le_mul (hsymbol i j) (hcoeff i j) (abs_nonneg _) hsymbolNonneg
    calc
      |mismatchTerm j|
        ≤ |reference.symbol i j * (input (approx.transport i j) - input (reference.transport i j))|
            + |(approx.symbol i j - reference.symbol i j) * input (approx.transport i j)| := by
              simpa [Real.norm_eq_abs, mismatchTerm] using
                norm_add_le
                  (reference.symbol i j * (input (approx.transport i j) - input (reference.transport i j)))
                  ((approx.symbol i j - reference.symbol i j) * input (approx.transport i j))
      _ ≤ referenceSymbolBound * transportError + symbolError * coeffBound := by
        linarith
  have hsumBound :
      ∑ j, |mismatchTerm j| ≤ ∑ _j : Fin K, pairBound := by
    refine Finset.sum_le_sum ?_
    intro j hj
    exact hterm j
  calc
    |(∑ j, mismatchTerm j) + approx.residual i|
      ≤ |∑ j, mismatchTerm j| + |approx.residual i| := by
          simpa [Real.norm_eq_abs] using
            norm_add_le
              (∑ j, mismatchTerm j)
              (approx.residual i)
    _ ≤ ∑ j, |mismatchTerm j| + |approx.residual i| := by
          gcongr
    _ ≤ ∑ _j : Fin K, pairBound + |approx.residual i| := by
          gcongr
    _ = (K : ℝ) * pairBound + |approx.residual i| := by
      simp [pairBound, add_comm, add_assoc, mul_add]
    _ ≤ (K : ℝ) * pairBound + truncationError := by
      linarith [hresidual i]

theorem l1_stencil_propagation_bound
    {n K : Nat}
    (input : Coeff n)
    (reference : StencilReferenceLayer n K)
    (approx : StencilApproxLayer n K)
    (transportError symbolError truncationError coeffBound referenceSymbolBound : ℝ)
    (htransportNonneg : 0 ≤ transportError)
    (hsymbolNonneg : 0 ≤ symbolError)
    (hcoeffNonneg : 0 ≤ coeffBound)
    (hrefNonneg : 0 ≤ referenceSymbolBound)
    (htransport :
      ∀ i j, |input (approx.transport i j) - input (reference.transport i j)| ≤ transportError)
    (hsymbol : ∀ i j, |approx.symbol i j - reference.symbol i j| ≤ symbolError)
    (hresidual : ∀ i, |approx.residual i| ≤ truncationError)
    (hcoeff : ∀ i j, |input (approx.transport i j)| ≤ coeffBound)
    (href : ∀ i j, |reference.symbol i j| ≤ referenceSymbolBound) :
    l1Norm (fun i => applyStencilApprox approx input i - applyStencilReference reference input i)
      ≤ n * ((K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) + truncationError) := by
  let bound := (K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) + truncationError
  have hsum :
      ∑ i, |applyStencilApprox approx input i - applyStencilReference reference input i|
        ≤ ∑ _i : Fin n, bound := by
    refine Finset.sum_le_sum ?_
    intro i hi
    exact
      pointwise_stencil_propagation_bound input reference approx transportError symbolError truncationError coeffBound referenceSymbolBound
        htransportNonneg hsymbolNonneg hcoeffNonneg hrefNonneg
        htransport hsymbol hresidual hcoeff href i
  calc
    l1Norm (fun i => applyStencilApprox approx input i - applyStencilReference reference input i)
      = ∑ i, |applyStencilApprox approx input i - applyStencilReference reference input i| := by rfl
    _ ≤ ∑ _i : Fin n, bound := hsum
    _ = n * bound := by
      simp [bound]
      ring

end Microlocal
end MiNO
