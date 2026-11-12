diagnosis D0001
  for: S0012, S0013, S0014, S0015, S0016, S0017, S0018
  misfit: KS-CONTROL-BASE-RATE
  line: V2 compares the probe's raw accuracy with the control task's raw accuracy, but type-level random labels at the class prior give a token-level positive rate that depends on which frequent types drew a 1; for a rare one-vs-rest class the control's own base rate can exceed the real one, so V2 fails without any evidence about the category (S0013: acc 0.942, base 0.926, control max 0.967)
  rule: compare excess over each label set's own majority rate, (acc - base) / (1 - base), for the real and the control fits, and record the control's token-level base rate beside it
  applies: verb in {probe}
  requires: control_base_rate
  signature: verb=probe & V1=true & V2=false & noun!=pos10
  test: probe pos-ovr:ADV at pre/mlp1 predict V2 = false
  author: ratchet-b9 (diagnoser)
