import Mathlib

namespace MiNO
namespace Microlocal

theorem shared_prediction_obstruction
    (prediction target₁ target₂ : ℝ) :
    |target₁ - target₂| / 2
      ≤ max |prediction - target₁| |prediction - target₂| := by
  have htriangle : |target₁ - target₂| ≤ |prediction - target₁| + |prediction - target₂| := by
    calc
      |target₁ - target₂| = |(target₁ - prediction) + (prediction - target₂)| := by ring_nf
      _ ≤ |target₁ - prediction| + |prediction - target₂| := by
        simpa [Real.norm_eq_abs] using norm_add_le (target₁ - prediction) (prediction - target₂)
      _ = |prediction - target₁| + |prediction - target₂| := by rw [abs_sub_comm]
  have hmax₁ : |prediction - target₁| ≤ max |prediction - target₁| |prediction - target₂| := by
    exact le_max_left _ _
  have hmax₂ : |prediction - target₂| ≤ max |prediction - target₁| |prediction - target₂| := by
    exact le_max_right _ _
  have hsum :
      |prediction - target₁| + |prediction - target₂|
        ≤ 2 * max |prediction - target₁| |prediction - target₂| := by
    nlinarith
  have hbound : |target₁ - target₂| ≤ 2 * max |prediction - target₁| |prediction - target₂| := by
    exact le_trans htriangle hsum
  have hnonneg : 0 ≤ max |prediction - target₁| |prediction - target₂| := by
    exact le_trans (abs_nonneg _) hmax₁
  nlinarith

theorem global_scalar_obstruction
    (c sharedInput target₁ target₂ : ℝ) :
    |target₁ - target₂| / 2
      ≤ max |c * sharedInput - target₁| |c * sharedInput - target₂| := by
  simpa using shared_prediction_obstruction (prediction := c * sharedInput) target₁ target₂

theorem packetwise_scalar_obstruction
    (c sharedInput symbol₁ symbol₂ : ℝ) :
    |symbol₁ * sharedInput - symbol₂ * sharedInput| / 2
      ≤ max |c * sharedInput - symbol₁ * sharedInput|
          |c * sharedInput - symbol₂ * sharedInput| := by
  simpa using
    global_scalar_obstruction
      (c := c)
      (sharedInput := sharedInput)
      (target₁ := symbol₁ * sharedInput)
      (target₂ := symbol₂ * sharedInput)

end Microlocal
end MiNO
