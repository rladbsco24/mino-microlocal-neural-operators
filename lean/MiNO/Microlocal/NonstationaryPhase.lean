import Mathlib
import MiNO.Microlocal.PacketMatrix
import MiNO.Microlocal.RestrictedBridge

namespace MiNO
namespace Microlocal

structure RestrictedNonstationaryPhaseCertificate
    {n : Nat}
    (M : PacketMatrix n)
    (shell : Fin n → Fin n → Nat)
    (K U : Nat) where
  count : ℕ → ℝ
  entry : ℕ → ℝ
  A : ℝ
  D : ℝ
  g : ℝ
  s : ℝ
  hA : 0 ≤ A
  hD : 0 ≤ D
  hg : 0 ≤ g
  hs : 0 ≤ s
  hgs : g * s < 1
  count_nonneg : ∀ ℓ, ℓ ∈ Finset.Ico K U → 0 ≤ count ℓ
  entry_nonneg : ∀ ℓ, ℓ ∈ Finset.Ico K U → 0 ≤ entry ℓ
  count_bound : ∀ ℓ, ℓ ∈ Finset.Ico K U → count ℓ ≤ A * g ^ ℓ
  entry_bound : ∀ ℓ, ℓ ∈ Finset.Ico K U → entry ℓ ≤ D * s ^ ℓ
  -- This is the finite certificate extracted from the classical oscillatory
  -- integral proof: each packet tail action is bounded by the certified shell
  -- tail times the input coefficient envelope.
  tail_apply_to_shell :
    ∀ (input : Coeff n) (coeffBound : ℝ),
      0 ≤ coeffBound →
      (∀ j, |input j| ≤ coeffBound) →
      ∀ i,
        |matrixTailApply M shell K input i| ≤
          coeffBound * shellTail (shellMassFrom count entry) K U

noncomputable def RestrictedNonstationaryPhaseCertificate.bound
    {n : Nat}
    {M : PacketMatrix n}
    {shell : Fin n → Fin n → Nat}
    {K U : Nat}
    (cert : RestrictedNonstationaryPhaseCertificate M shell K U) : ℝ :=
  restrictedOffGraphBound cert.A cert.D cert.g cert.s K

theorem restricted_nonstationary_phase_certificate_tail
    {n : Nat}
    {M : PacketMatrix n}
    {shell : Fin n → Fin n → Nat}
    {K U : Nat}
    (cert : RestrictedNonstationaryPhaseCertificate M shell K U)
    (input : Coeff n)
    (coeffBound : ℝ)
    (hcoeffNonneg : 0 ≤ coeffBound)
    (hcoeff : ∀ j, |input j| ≤ coeffBound) :
    ∀ i, |matrixTailApply M shell K input i| ≤ coeffBound * cert.bound := by
  have hshell :
      shellTail (shellMassFrom cert.count cert.entry) K U
        ≤ restrictedOffGraphBound cert.A cert.D cert.g cert.s K := by
    exact
      restricted_nonstationary_phase_bridge
        (A := cert.A)
        (D := cert.D)
        (g := cert.g)
        (s := cert.s)
        (K := K)
        (U := U)
        (count := cert.count)
        (entry := cert.entry)
        cert.hA
        cert.hD
        cert.hg
        cert.hs
        cert.hgs
        cert.count_nonneg
        cert.entry_nonneg
        cert.count_bound
        cert.entry_bound
  intro i
  have hpoint := cert.tail_apply_to_shell input coeffBound hcoeffNonneg hcoeff i
  exact le_trans hpoint (mul_le_mul_of_nonneg_left hshell hcoeffNonneg)

theorem packet_almost_diagonalization_l1_from_certificate
    {n : Nat}
    (M : PacketMatrix n)
    (shell : Fin n → Fin n → Nat)
    (K U : Nat)
    (input : Coeff n)
    (coeffBound : ℝ)
    (cert : RestrictedNonstationaryPhaseCertificate M shell K U)
    (hcoeffNonneg : 0 ≤ coeffBound)
    (hcoeff : ∀ j, |input j| ≤ coeffBound) :
    l1Norm (matrixTailApply M shell K input)
      ≤ n * (coeffBound * cert.bound) := by
  exact
    matrix_tail_l1_of_pointwise_bound
      (M := M)
      (shell := shell)
      (K := K)
      (input := input)
      (tailBound := coeffBound * cert.bound)
      (restricted_nonstationary_phase_certificate_tail
        (M := M)
        (shell := shell)
        (K := K)
        (U := U)
        cert
        input
        coeffBound
        hcoeffNonneg
        hcoeff)

end Microlocal
end MiNO
