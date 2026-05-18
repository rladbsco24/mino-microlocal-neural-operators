import Mathlib
import MiNO.Microlocal.AbstractSparsity
import MiNO.Microlocal.ResolutionTransfer

namespace MiNO
namespace Microlocal

theorem wave_transfer_rate_from_packet_tail
    {nCoarse nFine : Nat}
    (restrict : Coeff nFine → Coeff nCoarse)
    (prolong : Coeff nCoarse → Coeff nFine)
    (referenceCoarse : Coeff nCoarse → Coeff nCoarse)
    (referenceFine : Coeff nFine → Coeff nFine)
    (approxCoarse : Coeff nCoarse → Coeff nCoarse)
    (inputFine : Coeff nFine)
    (κ coarseError tailError coverError Ccoarse Ctransfer h : ℝ)
    (β α : Nat)
    (hκ : 0 ≤ κ)
    (hstable : l1Stable κ prolong)
    (hcoarse :
      l1Norm (fun i => approxCoarse (restrict inputFine) i - referenceCoarse (restrict inputFine) i)
        ≤ coarseError)
    (htransfer :
      l1Norm (fun i => prolong (referenceCoarse (restrict inputFine)) i - referenceFine inputFine i)
        ≤ tailError + coverError)
    (hcoarseRate : coarseError ≤ Ccoarse * h ^ β)
    (htransferRate : tailError + coverError ≤ Ctransfer * h ^ α) :
    l1Norm (fun i => prolong (approxCoarse (restrict inputFine)) i - referenceFine inputFine i)
      ≤ κ * (Ccoarse * h ^ β) + Ctransfer * h ^ α := by
  exact
    cross_resolution_transfer_rate
      (restrict := restrict)
      (prolong := prolong)
      (referenceCoarse := referenceCoarse)
      (referenceFine := referenceFine)
      (approxCoarse := approxCoarse)
      (inputFine := inputFine)
      (κ := κ)
      (coarseError := coarseError)
      (transferError := tailError + coverError)
      (Ccoarse := Ccoarse)
      (Ctransfer := Ctransfer)
      (h := h)
      (β := β)
      (α := α)
      hκ hstable hcoarse htransfer hcoarseRate htransferRate

end Microlocal
end MiNO
