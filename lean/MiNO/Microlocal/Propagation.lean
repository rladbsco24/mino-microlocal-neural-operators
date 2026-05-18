import Mathlib
import MiNO.Microlocal.FinitePacket

open scoped BigOperators

namespace MiNO
namespace Microlocal

theorem pointwise_propagation_bound
    {n : Nat}
    (input : Coeff n)
    (reference : ReferenceLayer n)
    (approx : ApproxLayer n)
    (transportError symbolError truncationError coeffBound referenceSymbolBound : ℝ)
    (htransport :
      ∀ i, |input (approx.transport i) - input (reference.transport i)| ≤ transportError)
    (hsymbol : ∀ i, |approx.symbol i - reference.symbol i| ≤ symbolError)
    (hresidual : ∀ i, |approx.residual i| ≤ truncationError)
    (hcoeff : ∀ i, |input (approx.transport i)| ≤ coeffBound)
    (href : ∀ i, |reference.symbol i| ≤ referenceSymbolBound) :
    ∀ i,
      |applyApprox approx input i - applyReference reference input i|
        ≤ referenceSymbolBound * transportError + symbolError * coeffBound + truncationError := by
  intro i
  have hnonnegTransport : 0 ≤ transportError := by
    nlinarith [abs_nonneg (input (approx.transport i) - input (reference.transport i)), htransport i]
  have hnonnegSymbol : 0 ≤ symbolError := by
    nlinarith [abs_nonneg (approx.symbol i - reference.symbol i), hsymbol i]
  have hnonnegResidual : 0 ≤ truncationError := by
    nlinarith [abs_nonneg (approx.residual i), hresidual i]
  have hnonnegCoeff : 0 ≤ coeffBound := by
    nlinarith [abs_nonneg (input (approx.transport i)), hcoeff i]
  have hnonnegReference : 0 ≤ referenceSymbolBound := by
    nlinarith [abs_nonneg (reference.symbol i), href i]
  have hdecomp :
      applyApprox approx input i - applyReference reference input i
        =
          reference.symbol i * (input (approx.transport i) - input (reference.transport i))
            + (approx.symbol i - reference.symbol i) * input (approx.transport i)
            + approx.residual i := by
    simp [applyApprox, applyReference]
    ring
  rw [hdecomp]
  have h₁ :
      |reference.symbol i * (input (approx.transport i) - input (reference.transport i))|
        ≤ referenceSymbolBound * transportError := by
    rw [abs_mul]
    exact mul_le_mul (href i) (htransport i) (abs_nonneg _) hnonnegReference
  have h₂ :
      |(approx.symbol i - reference.symbol i) * input (approx.transport i)|
        ≤ symbolError * coeffBound := by
    rw [abs_mul]
    exact mul_le_mul (hsymbol i) (hcoeff i) (abs_nonneg _) hnonnegSymbol
  have h₃ : |approx.residual i| ≤ truncationError := hresidual i
  let a := reference.symbol i * (input (approx.transport i) - input (reference.transport i))
  let b := (approx.symbol i - reference.symbol i) * input (approx.transport i)
  let c := approx.residual i
  have habc : |a + b + c| ≤ |a| + |b| + |c| := by
    have hab : |a + b| ≤ |a| + |b| := by
      simpa [Real.norm_eq_abs] using norm_add_le a b
    have habc' : |(a + b) + c| ≤ |a + b| + |c| := by
      simpa [Real.norm_eq_abs] using norm_add_le (a + b) c
    nlinarith
  calc
    |reference.symbol i * (input (approx.transport i) - input (reference.transport i))
        + (approx.symbol i - reference.symbol i) * input (approx.transport i)
        + approx.residual i|
      = |a + b + c| := by simp [a, b, c]
    _ ≤ |a| + |b| + |c| := habc
    _ ≤ referenceSymbolBound * transportError + symbolError * coeffBound + truncationError := by
      linarith

theorem l1_propagation_bound
    {n : Nat}
    (input : Coeff n)
    (reference : ReferenceLayer n)
    (approx : ApproxLayer n)
    (transportError symbolError truncationError coeffBound referenceSymbolBound : ℝ)
    (htransport :
      ∀ i, |input (approx.transport i) - input (reference.transport i)| ≤ transportError)
    (hsymbol : ∀ i, |approx.symbol i - reference.symbol i| ≤ symbolError)
    (hresidual : ∀ i, |approx.residual i| ≤ truncationError)
    (hcoeff : ∀ i, |input (approx.transport i)| ≤ coeffBound)
    (href : ∀ i, |reference.symbol i| ≤ referenceSymbolBound) :
    l1Norm (fun i => applyApprox approx input i - applyReference reference input i)
      ≤ n * (referenceSymbolBound * transportError + symbolError * coeffBound + truncationError) := by
  let bound := referenceSymbolBound * transportError + symbolError * coeffBound + truncationError
  have hsum :
      ∑ i, |applyApprox approx input i - applyReference reference input i|
        ≤ ∑ _i : Fin n, bound := by
    refine Finset.sum_le_sum ?_
    intro i hi
    exact pointwise_propagation_bound input reference approx transportError symbolError truncationError coeffBound referenceSymbolBound
      htransport hsymbol hresidual hcoeff href i
  calc
    l1Norm (fun i => applyApprox approx input i - applyReference reference input i)
      = ∑ i, |applyApprox approx input i - applyReference reference input i| := by rfl
    _ ≤ ∑ _i : Fin n, bound := hsum
    _ = n * bound := by
      simp [bound]
      ring

end Microlocal
end MiNO
