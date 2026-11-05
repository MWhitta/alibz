---
key: raissi2019pinn
title: "Physics-informed neural networks"
year: 2019
topic: ml
relevance: 1
---

**Method/claim**: Neural nets trained with PDE residuals as soft constraints; solves forward/inverse problems.

**Adoptable specifics**: Loss = data MSE + PDE residual at collocation points; automatic differentiation.

**Where it would enter alibz**: transformer_plan: physics-informed loss = reconstruction through the alibz forward model (Saha-Boltzmann line synthesis) of the predicted (T,n_e,c). Differentiable surrogate of synthetic.py needed.

**Fit-quality test**: synthetic suite S (bench/synth_cases.py) T error / log n_e error of the model with vs without forward-reconstruction loss.

**Caveats**: Forward model must be differentiable and cheap; PDE notion not used.
