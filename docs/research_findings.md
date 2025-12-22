# Research Findings (Dec 16, 2025)

## NinjaTrader commissions (MES micro futures)
- Published plan rates (per side): Free $0.39, Monthly $0.29, Lifetime $0.09; exchange/clearing/NFA fees extra. citeturn0search0  
- All-in per-side cost (includes exchange $0.37, clearing $0.19, NFA) from Feb 28, 2025 PDF: Lifetime ≈ $0.65, Monthly ≈ $0.81, Free ≈ $0.91 → round‑turn ≈ $1.30 / $1.62 / $1.82 respectively. citeturn0search5  
- Technology/routing fees can add $0.05–$0.25 per contract if using CQG/TT/third‑party bridges. citeturn0search1  
- For backtests, model commission as all‑in round‑turn: choose $1.60 baseline (Monthly) and make it configurable.

## Time encoding: cyclic sin/cos vs. adding an LSTM
- Cyclic encoding (sin/cos of hour) preserves adjacency (23:00 near 00:00), is model‑agnostic, cheap, and works well for tree/linear models. citeturn2search0turn2search5  
- LSTM advantages: captures longer, non‑periodic dependencies and regime shifts; multiple 2025 studies show LSTM variants outperform classical baselines on financial series. citeturn3academia12turn3search6  
- Costs/risks of LSTM: heavier training time (hours vs minutes), higher overfit risk on intraday noise, requires sequence windows and GPU; inference latency higher than simple MLP on cyclic features.  
- Recommended path: keep cyclic hour sin/cos as default features; add an optional temporal block (small LSTM/GRU or temporal conv) only if offline experiments show material uplift in Sharpe/IC; gate with a feature flag in Python training.

## Action space granularity for RL trading
- Discrete (flat/long/short) is stable and easy to constrain; limited sizing expressiveness.  
- Continuous sizing enables smoother risk and inventory control but must be clipped to margin/position limits and mapped to tick sizes; can be sensitive to slippage model. citeturn1search2  
- Hybrid approaches (“learn continuously, act discretely”) scope a continuous proposal then snap to discrete price/size grid, improving sample efficiency and stability in order‑execution tasks. citeturn1search0turn1search1  
- Guidance for this project: start with small discrete actions {flat, +1, –1 micro} for robustness; add a hybrid head that outputs desired size (continuous) then discretizes to tick/lot for limit orders once baseline is stable; keep slippage/commission in reward.
