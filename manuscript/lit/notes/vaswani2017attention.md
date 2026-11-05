---
key: vaswani2017attention
title: "Attention is all you need"
year: 2017
topic: ml
relevance: 1
---

**Method/claim**: Transformer: multi-head scaled dot-product attention, positional encodings, base d_model=512, 8 heads, 6 layers.

**Adoptable specifics**: Sec. 3: softmax(QK^T/sqrt(d_k))V; sinusoidal positions; label smoothing 0.1; Adam beta2=0.98, warmup 4000.

**Where it would enter alibz**: transformer_plan: input is a long (n_pixels~6k+) spectrum; vanilla attention O(n^2) is expensive: tokenise by peaks (peak list as set) rather than pixels, or patch (see he2022mae).

**Fit-quality test**: Model-level: accuracy vs token granularity on S.

**Caveats**: Architecture reference only.
