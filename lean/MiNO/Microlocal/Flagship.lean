import MiNO.Microlocal.FinitePacket

namespace MiNO
namespace Microlocal

/-!
Finite algebraic companions for the extended packet-matrix theorem package.

These lemmas do not formalize continuous stationary phase, Egorov theory, or
outgoing resolvent analysis.  They formalize the finite budget algebra used
after those classical estimates have produced packet matrices, thresholds, and
operator-norm envelopes.
-/

/-- A packet-microscope response is above threshold at `(i,j)`. -/
def superlevel {n : Nat} (response : Fin n → Fin n → ℝ) (τ : ℝ)
    (i j : Fin n) : Prop :=
  τ ≤ response i j

/--
If all off-tube packet microscope responses are below threshold, every
thresholded response is inside the tube.
-/
theorem thresholded_ridge_inside_tube
    {n : Nat}
    (response : Fin n → Fin n → ℝ)
    (tube : Fin n → Fin n → Prop)
    (τ : ℝ)
    (hOff : ∀ i j, ¬ tube i j → response i j < τ)
    {i j : Fin n}
    (hlev : superlevel response τ i j) :
    tube i j := by
  by_contra htube
  exact not_lt_of_ge hlev (hOff i j htube)

/--
If an active packet response is above an elliptic lower envelope and the
threshold is below that envelope, the active response is visible.
-/
theorem active_ridge_visible
    {n : Nat}
    (response : Fin n → Fin n → ℝ)
    (τ lower : ℝ)
    {i j : Fin n}
    (hLower : lower ≤ response i j)
    (hTau : τ ≤ lower) :
    superlevel response τ i j := by
  exact le_trans hTau hLower

/-- Additive fingerprint-stability budget. -/
theorem finite_fingerprint_budget_bound
    {eΛ ea eR EΛ Ea ER CΛ Ca CR : ℝ}
    (hΛ : eΛ ≤ EΛ)
    (ha : ea ≤ Ea)
    (hR : eR ≤ ER)
    (hCΛ : 0 ≤ CΛ)
    (hCa : 0 ≤ Ca)
    (hCR : 0 ≤ CR) :
    CΛ * eΛ + Ca * ea + CR * eR
      ≤ CΛ * EΛ + Ca * Ea + CR * ER := by
  have h1 : CΛ * eΛ ≤ CΛ * EΛ := by
    exact mul_le_mul_of_nonneg_left hΛ hCΛ
  have h2 : Ca * ea ≤ Ca * Ea := by
    exact mul_le_mul_of_nonneg_left ha hCa
  have h3 : CR * eR ≤ CR * ER := by
    exact mul_le_mul_of_nonneg_left hR hCR
  nlinarith

/--
Finite Egorov budget expansion:
`||M^* A M - M₀^* A M₀||` is controlled by the two linear perturbation terms
and the quadratic perturbation term once `||M-M₀|| ≤ eps`.
-/
theorem finite_egorov_budget_identity
    (A M eps : ℝ) :
    A * eps * (2 * M + eps) = 2 * eps * A * M + eps * eps * A := by
  ring

/-- Finite composition budget for two packet laws. -/
theorem finite_composition_budget_bound
    {L1 L2 e1 e2 E1 E2 Ec : ℝ}
    (hL1 : 0 ≤ L1)
    (hL2 : 0 ≤ L2)
    (he2 : 0 ≤ e2)
    (hE1 : 0 ≤ E1)
    (h1 : e1 ≤ E1)
    (h2 : e2 ≤ E2) :
    L2 * e1 + L1 * e2 + e1 * e2 + Ec
      ≤ L2 * E1 + L1 * E2 + E1 * E2 + Ec := by
  have hlin1 : L2 * e1 ≤ L2 * E1 := by
    exact mul_le_mul_of_nonneg_left h1 hL2
  have hlin2 : L1 * e2 ≤ L1 * E2 := by
    exact mul_le_mul_of_nonneg_left h2 hL1
  have hprod : e1 * e2 ≤ E1 * E2 := by
    exact mul_le_mul h1 h2 he2 hE1
  nlinarith

/--
If a theorem budget is certified by a field loss and a proxy penalty, any
generalization/optimization certificate for those quantities transfers to the
theorem budget.
-/
theorem finite_learning_budget_certificate
    {Bthm Lhat Linf gen opt Pi c0 c1 : ℝ}
    (hc0 : 0 ≤ c0)
    (hLoss : Lhat ≤ Linf + gen + opt)
    (hBudget : Bthm ≤ c0 * Lhat + c1 * Pi) :
    Bthm ≤ c0 * (Linf + gen + opt) + c1 * Pi := by
  have hscaled : c0 * Lhat ≤ c0 * (Linf + gen + opt) := by
    exact mul_le_mul_of_nonneg_left hLoss hc0
  nlinarith

end Microlocal
end MiNO
