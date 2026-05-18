import Mathlib
import MiNO.Microlocal.PacketTail
import MiNO.Microlocal.ShellCount

namespace MiNO
namespace Microlocal

def shellMassFrom (count entry : ℕ → ℝ) : ℕ → ℝ :=
  fun ℓ => count ℓ * entry ℓ

theorem abstract_packet_sparsity_tail
    (A D g s : ℝ)
    (L U : ℕ)
    (count entry : ℕ → ℝ)
    (hA : 0 ≤ A)
    (hD : 0 ≤ D)
    (hg : 0 ≤ g)
    (hs : 0 ≤ s)
    (hgs : g * s < 1)
    (hcountNonneg : ∀ i, i ∈ Finset.Ico L U → 0 ≤ count i)
    (hentryNonneg : ∀ i, i ∈ Finset.Ico L U → 0 ≤ entry i)
    (hcount : ∀ i, i ∈ Finset.Ico L U → count i ≤ A * g ^ i)
    (hentry : ∀ i, i ∈ Finset.Ico L U → entry i ≤ D * s ^ i) :
    shellTail (shellMassFrom count entry) L U ≤ (A * D) * ((g * s) ^ L / (1 - g * s)) := by
  simpa [shellMassFrom] using
    counted_geometric_shell_tail_bound
      (A := A) (D := D) (g := g) (s := s) (L := L) (U := U)
      (count := count) (entry := entry) (shellMass := shellMassFrom count entry)
      hA hD hg hs hgs
      (by intro i; rfl)
      hcountNonneg hentryNonneg hcount hentry

theorem canonical_tube_truncation_bound
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
    shellTail (shellMassFrom count entry) K U ≤ (A * D) * ((g * s) ^ K / (1 - g * s)) := by
  exact
    abstract_packet_sparsity_tail
      (A := A) (D := D) (g := g) (s := s) (L := K) (U := U)
      (count := count) (entry := entry)
      hA hD hg hs hgs hcountNonneg hentryNonneg hcount hentry

end Microlocal
end MiNO
