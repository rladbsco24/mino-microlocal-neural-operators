import Mathlib
import MiNO.Microlocal.StencilPropagation
import MiNO.Microlocal.WaveTransfer

namespace MiNO
namespace Microlocal

theorem bounded_stencil_transfer_bound
    {nCoarse nFine K : Nat}
    (restrict : Coeff nFine → Coeff nCoarse)
    (prolong : Coeff nCoarse → Coeff nFine)
    (referenceStencil : StencilReferenceLayer nCoarse K)
    (approxStencil : StencilApproxLayer nCoarse K)
    (referenceFine : Coeff nFine → Coeff nFine)
    (inputFine : Coeff nFine)
    (κ transportError symbolError truncationError coeffBound referenceSymbolBound transferError : ℝ)
    (hκ : 0 ≤ κ)
    (htransportNonneg : 0 ≤ transportError)
    (hsymbolNonneg : 0 ≤ symbolError)
    (hcoeffNonneg : 0 ≤ coeffBound)
    (hrefNonneg : 0 ≤ referenceSymbolBound)
    (hstable : l1Stable κ prolong)
    (htransport :
      ∀ i j,
        |(restrict inputFine) (approxStencil.transport i j) -
            (restrict inputFine) (referenceStencil.transport i j)| ≤ transportError)
    (hsymbol : ∀ i j, |approxStencil.symbol i j - referenceStencil.symbol i j| ≤ symbolError)
    (hresidual : ∀ i, |approxStencil.residual i| ≤ truncationError)
    (hcoeff : ∀ i j, |(restrict inputFine) (approxStencil.transport i j)| ≤ coeffBound)
    (href : ∀ i j, |referenceStencil.symbol i j| ≤ referenceSymbolBound)
    (htransfer :
      l1Norm
          (fun i =>
            prolong (applyStencilReference referenceStencil (restrict inputFine)) i -
              referenceFine inputFine i) ≤
        transferError) :
    l1Norm
        (fun i =>
          prolong (applyStencilApprox approxStencil (restrict inputFine)) i -
            referenceFine inputFine i) ≤
      κ *
          (nCoarse *
            ((K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) +
              truncationError)) +
        transferError := by
  have hcoarse :
      l1Norm
          (fun i =>
            applyStencilApprox approxStencil (restrict inputFine) i -
              applyStencilReference referenceStencil (restrict inputFine) i) ≤
        nCoarse *
          ((K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) +
            truncationError) := by
    exact
      l1_stencil_propagation_bound
        (input := restrict inputFine)
        (reference := referenceStencil)
        (approx := approxStencil)
        (transportError := transportError)
        (symbolError := symbolError)
        (truncationError := truncationError)
        (coeffBound := coeffBound)
        (referenceSymbolBound := referenceSymbolBound)
        htransportNonneg hsymbolNonneg hcoeffNonneg hrefNonneg
        htransport hsymbol hresidual hcoeff href
  exact
    cross_resolution_transfer_bound
      (restrict := restrict)
      (prolong := prolong)
      (referenceCoarse := fun a => applyStencilReference referenceStencil a)
      (referenceFine := referenceFine)
      (approxCoarse := fun a => applyStencilApprox approxStencil a)
      (inputFine := inputFine)
      (κ := κ)
      (coarseError :=
        nCoarse *
          ((K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) +
            truncationError))
      (transferError := transferError)
      hκ hstable hcoarse htransfer

theorem bounded_stencil_wave_transfer_rate
    {nCoarse nFine K : Nat}
    (restrict : Coeff nFine → Coeff nCoarse)
    (prolong : Coeff nCoarse → Coeff nFine)
    (referenceStencil : StencilReferenceLayer nCoarse K)
    (approxStencil : StencilApproxLayer nCoarse K)
    (referenceFine : Coeff nFine → Coeff nFine)
    (inputFine : Coeff nFine)
    (κ transportError symbolError truncationError coeffBound referenceSymbolBound : ℝ)
    (tailError coverError Ccoarse Ctransfer h : ℝ)
    (β α : Nat)
    (hκ : 0 ≤ κ)
    (htransportNonneg : 0 ≤ transportError)
    (hsymbolNonneg : 0 ≤ symbolError)
    (hcoeffNonneg : 0 ≤ coeffBound)
    (hrefNonneg : 0 ≤ referenceSymbolBound)
    (hstable : l1Stable κ prolong)
    (htransport :
      ∀ i j,
        |(restrict inputFine) (approxStencil.transport i j) -
            (restrict inputFine) (referenceStencil.transport i j)| ≤ transportError)
    (hsymbol : ∀ i j, |approxStencil.symbol i j - referenceStencil.symbol i j| ≤ symbolError)
    (hresidual : ∀ i, |approxStencil.residual i| ≤ truncationError)
    (hcoeff : ∀ i j, |(restrict inputFine) (approxStencil.transport i j)| ≤ coeffBound)
    (href : ∀ i j, |referenceStencil.symbol i j| ≤ referenceSymbolBound)
    (htransfer :
      l1Norm
          (fun i =>
            prolong (applyStencilReference referenceStencil (restrict inputFine)) i -
              referenceFine inputFine i) ≤
        tailError + coverError)
    (hcoarseRate :
      nCoarse *
          ((K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) +
            truncationError) ≤
        Ccoarse * h ^ β)
    (htransferRate : tailError + coverError ≤ Ctransfer * h ^ α) :
    l1Norm
        (fun i =>
          prolong (applyStencilApprox approxStencil (restrict inputFine)) i -
            referenceFine inputFine i) ≤
      κ * (Ccoarse * h ^ β) + Ctransfer * h ^ α := by
  have hcoarse :
      l1Norm
          (fun i =>
            applyStencilApprox approxStencil (restrict inputFine) i -
              applyStencilReference referenceStencil (restrict inputFine) i) ≤
        nCoarse *
          ((K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) +
            truncationError) := by
    exact
      l1_stencil_propagation_bound
        (input := restrict inputFine)
        (reference := referenceStencil)
        (approx := approxStencil)
        (transportError := transportError)
        (symbolError := symbolError)
        (truncationError := truncationError)
        (coeffBound := coeffBound)
        (referenceSymbolBound := referenceSymbolBound)
        htransportNonneg hsymbolNonneg hcoeffNonneg hrefNonneg
        htransport hsymbol hresidual hcoeff href
  exact
    wave_transfer_rate_from_packet_tail
      (restrict := restrict)
      (prolong := prolong)
      (referenceCoarse := fun a => applyStencilReference referenceStencil a)
      (referenceFine := referenceFine)
      (approxCoarse := fun a => applyStencilApprox approxStencil a)
      (inputFine := inputFine)
      (κ := κ)
      (coarseError :=
        nCoarse *
          ((K : ℝ) * (referenceSymbolBound * transportError + symbolError * coeffBound) +
            truncationError))
      (tailError := tailError)
      (coverError := coverError)
      (Ccoarse := Ccoarse)
      (Ctransfer := Ctransfer)
      (h := h)
      (β := β)
      (α := α)
      hκ hstable hcoarse htransfer hcoarseRate htransferRate

end Microlocal
end MiNO
