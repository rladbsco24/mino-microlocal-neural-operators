import Mathlib
import MiNO.Microlocal.FinitePacket

namespace MiNO
namespace Microlocal

def composeReference {n : Nat} (outer inner : ReferenceLayer n) : ReferenceLayer n where
  transport i := inner.transport (outer.transport i)
  symbol i := outer.symbol i * inner.symbol (outer.transport i)

theorem applyReference_compose
    {n : Nat}
    (outer inner : ReferenceLayer n)
    (input : Coeff n) :
    applyReference (composeReference outer inner) input
      =
      applyReference outer (applyReference inner input) := by
  funext i
  simp [applyReference, composeReference]
  ring

theorem applyApprox_identity_transport
    {n : Nat}
    (approx : ApproxLayer n)
    (input : Coeff n)
    (htransport : approx.transport = id) :
    applyApprox approx input
      =
      fun i => approx.symbol i * input i + approx.residual i := by
  funext i
  simp [applyApprox, htransport]

theorem applyApprox_identity_transport_zero_residual
    {n : Nat}
    (approx : ApproxLayer n)
    (input : Coeff n)
    (htransport : approx.transport = id)
    (hresidual : approx.residual = 0) :
    applyApprox approx input = fun i => approx.symbol i * input i := by
  funext i
  simp [applyApprox, htransport, hresidual]

theorem applyReference_identity_transport
    {n : Nat}
    (reference : ReferenceLayer n)
    (input : Coeff n)
    (htransport : reference.transport = id) :
    applyReference reference input = fun i => reference.symbol i * input i := by
  funext i
  simp [applyReference, htransport]

end Microlocal
end MiNO
