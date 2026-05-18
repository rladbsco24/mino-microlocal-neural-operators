import Mathlib

namespace MiNO
namespace Microlocal

def canonicalTube {n : Nat} (shell : Fin n → Nat) (K : Nat) : Finset (Fin n) :=
  Finset.univ.filter fun i => shell i < K

def canonicalShell {n : Nat} (shell : Fin n → Nat) (ℓ : Nat) : Finset (Fin n) :=
  Finset.univ.filter fun i => shell i = ℓ

theorem mem_canonicalTube {n : Nat} {shell : Fin n → Nat} {K : Nat} {i : Fin n} :
    i ∈ canonicalTube shell K ↔ shell i < K := by
  simp [canonicalTube]

theorem mem_canonicalShell {n : Nat} {shell : Fin n → Nat} {ℓ : Nat} {i : Fin n} :
    i ∈ canonicalShell shell ℓ ↔ shell i = ℓ := by
  simp [canonicalShell]

theorem canonicalShell_subset_tube
    {n : Nat}
    (shell : Fin n → Nat)
    {ℓ K : Nat}
    (hℓ : ℓ < K) :
    canonicalShell shell ℓ ⊆ canonicalTube shell K := by
  intro i hi
  rw [mem_canonicalShell] at hi
  rw [mem_canonicalTube]
  simpa [hi] using hℓ

theorem not_mem_tube_of_shell_ge
    {n : Nat}
    (shell : Fin n → Nat)
    {ℓ K : Nat}
    (hK : K ≤ ℓ)
    {i : Fin n}
    (hi : i ∈ canonicalShell shell ℓ) :
    i ∉ canonicalTube shell K := by
  rw [mem_canonicalShell] at hi
  intro hit
  rw [mem_canonicalTube] at hit
  have : ℓ < K := by simpa [hi] using hit
  exact Nat.not_lt.mpr hK this

theorem canonicalTube_mono
    {n : Nat}
    (shell : Fin n → Nat)
    {K L : Nat}
    (hKL : K ≤ L) :
    canonicalTube shell K ⊆ canonicalTube shell L := by
  intro i hi
  rw [mem_canonicalTube] at hi ⊢
  exact lt_of_lt_of_le hi hKL

end Microlocal
end MiNO
