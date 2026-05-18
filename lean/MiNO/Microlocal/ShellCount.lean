import Mathlib
import MiNO.Microlocal.CanonicalTube

namespace MiNO
namespace Microlocal

def shellCard {n : Nat} (shell : Fin n → Nat) (ℓ : Nat) : Nat :=
  (canonicalShell shell ℓ).card

def geometricShellCount {n : Nat} (shell : Fin n → Nat) (A g : ℝ) : Prop :=
  ∀ ℓ, ((shellCard shell ℓ : Nat) : ℝ) ≤ A * g ^ ℓ

theorem shellCard_nonneg
    {n : Nat}
    (shell : Fin n → Nat)
    (ℓ : Nat) :
    0 ≤ (((shellCard shell ℓ : Nat) : ℝ)) := by
  positivity

theorem shellCard_bound_of_geometric
    {n : Nat}
    (shell : Fin n → Nat)
    (A g : ℝ)
    (hgeom : geometricShellCount shell A g) :
    ∀ ℓ, (((shellCard shell ℓ : Nat) : ℝ)) ≤ A * g ^ ℓ := by
  intro ℓ
  exact hgeom ℓ

theorem shellCard_zero_when_empty
    {n : Nat}
    (shell : Fin n → Nat)
    (ℓ : Nat)
    (hempty : canonicalShell shell ℓ = ∅) :
    (((shellCard shell ℓ : Nat) : ℝ)) = 0 := by
  simp [shellCard, hempty]

end Microlocal
end MiNO
