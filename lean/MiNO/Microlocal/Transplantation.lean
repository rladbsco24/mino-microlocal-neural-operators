import Mathlib
import MiNO.Microlocal.Approximation

namespace MiNO
namespace Microlocal

structure BudgetWitness
    {n : Nat}
    (input : Coeff n)
    (reference : ReferenceLayer n)
    (approx : ApproxLayer n) where
  transportError : ℝ
  symbolError : ℝ
  truncationError : ℝ
  coeffBound : ℝ
  referenceSymbolBound : ℝ
  htransport :
    ∀ i, |input (approx.transport i) - input (reference.transport i)| ≤ transportError
  hsymbol : ∀ i, |approx.symbol i - reference.symbol i| ≤ symbolError
  hresidual : ∀ i, |approx.residual i| ≤ truncationError
  hcoeff : ∀ i, |input (approx.transport i)| ≤ coeffBound
  href : ∀ i, |reference.symbol i| ≤ referenceSymbolBound

def BudgetWitness.totalBudget
    {n : Nat}
    {input : Coeff n}
    {reference : ReferenceLayer n}
    {approx : ApproxLayer n}
    (witness : BudgetWitness input reference approx) : ℝ :=
  witness.referenceSymbolBound * witness.transportError
    + witness.symbolError * witness.coeffBound
    + witness.truncationError

theorem semidiscrete_transplantation_pointwise
    {n : Nat}
    (input : Coeff n)
    (reference : ReferenceLayer n)
    (approx : ApproxLayer n)
    (witness : BudgetWitness input reference approx) :
    ∀ i,
      |applyApprox approx input i - applyReference reference input i|
        ≤ witness.totalBudget := by
  intro i
  simpa [BudgetWitness.totalBudget] using
    pointwise_propagation_bound
      (input := input)
      (reference := reference)
      (approx := approx)
      (transportError := witness.transportError)
      (symbolError := witness.symbolError)
      (truncationError := witness.truncationError)
      (coeffBound := witness.coeffBound)
      (referenceSymbolBound := witness.referenceSymbolBound)
      witness.htransport witness.hsymbol witness.hresidual witness.hcoeff witness.href i

theorem semidiscrete_transplantation_l1
    {n : Nat}
    (input : Coeff n)
    (reference : ReferenceLayer n)
    (approx : ApproxLayer n)
    (witness : BudgetWitness input reference approx) :
    l1Norm (fun i => applyApprox approx input i - applyReference reference input i)
      ≤ n * witness.totalBudget := by
  simpa [BudgetWitness.totalBudget] using
    l1_propagation_bound
      (input := input)
      (reference := reference)
      (approx := approx)
      (transportError := witness.transportError)
      (symbolError := witness.symbolError)
      (truncationError := witness.truncationError)
      (coeffBound := witness.coeffBound)
      (referenceSymbolBound := witness.referenceSymbolBound)
      witness.htransport witness.hsymbol witness.hresidual witness.hcoeff witness.href

theorem semidiscrete_transplantation_approximation
    {n : Nat}
    (input : Coeff n)
    (reference : ReferenceLayer n)
    (approx : ApproxLayer n)
    (witness : BudgetWitness input reference approx)
    (ε : ℝ)
    (hbudget : n * witness.totalBudget ≤ ε) :
    l1Norm (fun i => applyApprox approx input i - applyReference reference input i) ≤ ε := by
  simpa [BudgetWitness.totalBudget] using
    finite_approximation_corollary
      (input := input)
      (reference := reference)
      (approx := approx)
      (transportError := witness.transportError)
      (symbolError := witness.symbolError)
      (truncationError := witness.truncationError)
      (coeffBound := witness.coeffBound)
      (referenceSymbolBound := witness.referenceSymbolBound)
      (ε := ε)
      witness.htransport witness.hsymbol witness.hresidual witness.hcoeff witness.href hbudget

end Microlocal
end MiNO
