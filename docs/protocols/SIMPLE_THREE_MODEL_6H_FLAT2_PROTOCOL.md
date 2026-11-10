# Simple three-model 6h — 2% flat-band follow-up

Frozen on 2026-08-23 after the preregistered `±0.20%` pilot returned
NO EDGE and before this follow-up was trained.

The entire original protocol remains unchanged except:

- FLAT band is fixed at `±2.00%`;
- `UP=1` only above `+2.00%` over six hours;
- `DOWN=1` only below `-2.00%`;
- no intermediate bands are tested;
- output is stored separately under
  `causal_flip/reports/simple_three_model_6h_flat2/`.

Expected train class distribution measured before training:

- DOWN `14.6%`;
- FLAT `69.2%`;
- UP `16.2%`.

This is an explicitly post-hoc follow-up. Continue to Optuna only if the same
raw edge criteria from `SIMPLE_THREE_MODEL_6H_PROTOCOL.md` pass.
