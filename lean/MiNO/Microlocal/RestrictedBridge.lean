import Mathlib
import MiNO.Microlocal.AbstractSparsity
import MiNO.Microlocal.ResolutionTransfer

namespace MiNO
namespace Microlocal

noncomputable def restrictedOffGraphBound (A D g s : ℝ) (K : ℕ) : ℝ :=
  (A * D) * ((g * s) ^ K / (1 - g * s))

theorem restricted_nonstationary_phase_bridge
    (A D g s : ℝ)
    (K U : ℕ)
    (count entry : ℕ → ℝ)
    (hA : 0 ≤ A)
    (hD : 0 ≤ D)
    (hg : 0 ≤ g)
    (hs : 0 ≤ s)
    (hgs : g * s < 1)
    (hcountNonneg : ∀ i, i ∈ Finset.Ico K U → 0 ≤ count i)
    (hentryNonneg : ∀ i, i ∈ Finset.Ico K U → 0 ≤ entry i)
    (hcount : ∀ i, i ∈ Finset.Ico K U → count i ≤ A * g ^ i)
    (hentry : ∀ i, i ∈ Finset.Ico K U → entry i ≤ D * s ^ i) :
    shellTail (shellMassFrom count entry) K U ≤ restrictedOffGraphBound A D g s K := by
  simpa [restrictedOffGraphBound] using
    canonical_tube_truncation_bound
      (A := A)
      (D := D)
      (g := g)
      (s := s)
      (K := K)
      (U := U)
      (count := count)
      (entry := entry)
      hA
      hD
      hg
      hs
      hgs
      hcountNonneg
      hentryNonneg
      hcount
      hentry

theorem restricted_packet_almost_diagonalization_bridge
    {nCoarse nFine : Nat}
    (restrict : Coeff nFine → Coeff nCoarse)
    (prolong : Coeff nCoarse → Coeff nFine)
    (referenceCoarse : Coeff nCoarse → Coeff nCoarse)
    (referenceFine : Coeff nFine → Coeff nFine)
    (approxCoarse : Coeff nCoarse → Coeff nCoarse)
    (inputFine : Coeff nFine)
    (A D g s : ℝ)
    (K : ℕ)
    (κ coarseError tailError coverError : ℝ)
    (hκ : 0 ≤ κ)
    (hstable : l1Stable κ prolong)
    (hcoarse :
      l1Norm (fun i => approxCoarse (restrict inputFine) i - referenceCoarse (restrict inputFine) i)
        ≤ coarseError)
    (htransfer :
      l1Norm (fun i => prolong (referenceCoarse (restrict inputFine)) i - referenceFine inputFine i)
        ≤ tailError + coverError)
    (htail : tailError ≤ restrictedOffGraphBound A D g s K) :
    l1Norm (fun i => prolong (approxCoarse (restrict inputFine)) i - referenceFine inputFine i)
      ≤ κ * coarseError + (restrictedOffGraphBound A D g s K + coverError) := by
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
      (transferError := tailError + coverError)
      hκ
      hstable
      hcoarse
      htransfer
  linarith

end Microlocal
end MiNO
