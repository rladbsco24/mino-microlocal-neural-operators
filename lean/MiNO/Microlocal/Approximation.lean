import Mathlib
import MiNO.Microlocal.Propagation

namespace MiNO
namespace Microlocal

theorem finite_approximation_corollary
    {n : Nat}
    (input : Coeff n)
    (reference : ReferenceLayer n)
    (approx : ApproxLayer n)
    (transportError symbolError truncationError coeffBound referenceSymbolBound ε : ℝ)
    (htransport :
      ∀ i, |input (approx.transport i) - input (reference.transport i)| ≤ transportError)
    (hsymbol : ∀ i, |approx.symbol i - reference.symbol i| ≤ symbolError)
    (hresidual : ∀ i, |approx.residual i| ≤ truncationError)
    (hcoeff : ∀ i, |input (approx.transport i)| ≤ coeffBound)
    (href : ∀ i, |reference.symbol i| ≤ referenceSymbolBound)
    (hbudget :
      n * (referenceSymbolBound * transportError + symbolError * coeffBound + truncationError) ≤ ε) :
    l1Norm (fun i => applyApprox approx input i - applyReference reference input i) ≤ ε := by
  exact le_trans
    (l1_propagation_bound input reference approx transportError symbolError truncationError coeffBound referenceSymbolBound
      htransport hsymbol hresidual hcoeff href)
    hbudget

end Microlocal
end MiNO
