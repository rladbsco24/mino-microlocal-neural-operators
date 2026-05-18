import Mathlib

open scoped BigOperators

namespace MiNO
namespace Microlocal

abbrev Coeff (n : Nat) := Fin n → ℝ

structure PacketState (n : Nat) where
  coeff : Coeff n
  posX : Fin n → ℝ
  posY : Fin n → ℝ
  freqX : Fin n → ℝ
  freqY : Fin n → ℝ
  scale : Fin n → ℝ

structure ReferenceLayer (n : Nat) where
  transport : Fin n → Fin n
  symbol : Fin n → ℝ

structure ApproxLayer (n : Nat) where
  transport : Fin n → Fin n
  symbol : Fin n → ℝ
  residual : Fin n → ℝ

def applyReference {n : Nat} (layer : ReferenceLayer n) (input : Coeff n) : Coeff n :=
  fun i => layer.symbol i * input (layer.transport i)

def applyApprox {n : Nat} (layer : ApproxLayer n) (input : Coeff n) : Coeff n :=
  fun i => layer.symbol i * input (layer.transport i) + layer.residual i

def l1Norm {n : Nat} (coeff : Coeff n) : ℝ :=
  ∑ i, |coeff i|

end Microlocal
end MiNO
