import Mathlib

open scoped BigOperators

namespace MiNO
namespace Microlocal

def shellTail (shellMass : ℕ → ℝ) (L U : ℕ) : ℝ :=
  Finset.sum (Finset.Ico L U) shellMass

theorem geometric_shell_tail_bound
    (ρ C : ℝ)
    (L U : ℕ)
    (shellMass : ℕ → ℝ)
    (hρ0 : 0 ≤ ρ)
    (hρ1 : ρ < 1)
    (hC : 0 ≤ C)
    (hshell : ∀ i, i ∈ Finset.Ico L U → shellMass i ≤ C * ρ ^ i) :
    shellTail shellMass L U ≤ C * (ρ ^ L / (1 - ρ)) := by
  unfold shellTail
  have hsum :
      Finset.sum (Finset.Ico L U) shellMass ≤ Finset.sum (Finset.Ico L U) (fun i => C * ρ ^ i) := by
    refine Finset.sum_le_sum ?_
    intro i hi
    exact hshell i hi
  calc
    Finset.sum (Finset.Ico L U) shellMass
      ≤ Finset.sum (Finset.Ico L U) (fun i => C * ρ ^ i) := hsum
    _ = C * Finset.sum (Finset.Ico L U) (fun i => ρ ^ i) := by
      rw [Finset.mul_sum]
    _ ≤ C * (ρ ^ L / (1 - ρ)) := by
      exact mul_le_mul_of_nonneg_left (geom_sum_Ico_le_of_lt_one hρ0 hρ1) hC

theorem shell_mass_bound_from_count_and_decay
    (A D g s : ℝ)
    (ℓ : ℕ)
    (hA : 0 ≤ A)
    (_hD : 0 ≤ D)
    (hg : 0 ≤ g)
    (_hs : 0 ≤ s)
    (countBound : ℝ)
    (entryBound : ℝ)
    (_hcountNonneg : 0 ≤ countBound)
    (hentryNonneg : 0 ≤ entryBound)
    (hcount : countBound ≤ A * g ^ ℓ)
    (hentry : entryBound ≤ D * s ^ ℓ) :
    countBound * entryBound ≤ (A * D) * (g * s) ^ ℓ := by
  have hcount' : countBound * entryBound ≤ (A * g ^ ℓ) * (D * s ^ ℓ) := by
    exact mul_le_mul hcount hentry hentryNonneg (mul_nonneg hA (pow_nonneg hg _))
  calc
    countBound * entryBound ≤ (A * g ^ ℓ) * (D * s ^ ℓ) := hcount'
    _ = (A * D) * (g * s) ^ ℓ := by
      rw [mul_pow]
      ring_nf

theorem counted_geometric_shell_tail_bound
    (A D g s : ℝ)
    (L U : ℕ)
    (count entry shellMass : ℕ → ℝ)
    (hA : 0 ≤ A)
    (hD : 0 ≤ D)
    (hg : 0 ≤ g)
    (hs : 0 ≤ s)
    (hgs : g * s < 1)
    (hshellDef : ∀ i, shellMass i = count i * entry i)
    (hcountNonneg : ∀ i, i ∈ Finset.Ico L U → 0 ≤ count i)
    (hentryNonneg : ∀ i, i ∈ Finset.Ico L U → 0 ≤ entry i)
    (hcount : ∀ i, i ∈ Finset.Ico L U → count i ≤ A * g ^ i)
    (hentry : ∀ i, i ∈ Finset.Ico L U → entry i ≤ D * s ^ i) :
    shellTail shellMass L U ≤ (A * D) * ((g * s) ^ L / (1 - g * s)) := by
  apply geometric_shell_tail_bound (ρ := g * s) (C := A * D) (L := L) (U := U) (shellMass := shellMass)
  · positivity
  · exact hgs
  · exact mul_nonneg hA hD
  · intro i hi
    rw [hshellDef i]
    exact shell_mass_bound_from_count_and_decay
      (A := A) (D := D) (g := g) (s := s) (ℓ := i)
      hA hD hg hs (count i) (entry i) (hcountNonneg i hi) (hentryNonneg i hi) (hcount i hi) (hentry i hi)

end Microlocal
end MiNO
