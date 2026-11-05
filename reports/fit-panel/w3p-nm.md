# Fit-quality comparison: w3p-base -> w3p-nm

baseline config: `{}`  
variant config: `{"outer_polish": "nelder-mead"}`

## Per-item

| panel | item | metric | before | after | delta | verdict |
|---|---|---|---|---|---|---|
| synthetic | binary_ca_mg | median_rel_err | 0.1666 | 0.02046 | -0.1461 | better |
| synthetic | binary_ca_mg | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | abs_T_rel_err | 0.1117 | 0.02223 | -0.08942 | better |
| synthetic | binary_ca_mg | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | dominant_correct | 1 | 1 | 0 | same |
| synthetic | silicate_4el | median_rel_err | 0.3615 | 0.351 | -0.01052 | better |
| synthetic | silicate_4el | spurious_mass | 0 | 0 | 0 | same |
| synthetic | silicate_4el | abs_T_rel_err | 0.1048 | 0.07889 | -0.02596 | better |
| synthetic | silicate_4el | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | silicate_4el | dominant_correct | 1 | 1 | 0 | same |
| synthetic | feldspar_k_na | median_rel_err | 0.08967 | 0.2586 | 0.1689 | worse |
| synthetic | feldspar_k_na | spurious_mass | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | abs_T_rel_err | 0.01215 | 0.2095 | 0.1973 | worse |
| synthetic | feldspar_k_na | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | dominant_correct | 1 | 1 | 0 | same |
| synthetic | iron_rich_low_t | median_rel_err | 0.009964 | 0.01597 | 0.006001 | worse |
| synthetic | iron_rich_low_t | spurious_mass | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | abs_T_rel_err | 0.001947 | 0.01818 | 0.01624 | worse |
| synthetic | iron_rich_low_t | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | dominant_correct | 1 | 1 | 0 | same |
| synthetic | binary_argon_ambient | median_rel_err | 0.152 | 0.1766 | 0.02458 | worse |
| synthetic | binary_argon_ambient | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | abs_T_rel_err | 0.1125 | 0.02877 | -0.08374 | better |
| synthetic | binary_argon_ambient | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | dominant_correct | 1 | 1 | 0 | same |
| real | REE_01 | r_squared | -0.2883 | -0.286 | 0.002308 | better |
| real | REE_01 | median_stage_disagreement | 0.4263 | 0.4887 | 0.0624 | worse |
| real | REE_01 | n_detected | 10 | 10 | 0 | same |
| real | REE_01 | ne_at_bound | 1 | 0 | -1 | better |
| real | REE_44 | r_squared | 0.6625 | 0.6705 | 0.007994 | better |
| real | REE_44 | median_stage_disagreement | 0.4737 | 0.4638 | -0.009961 | better |
| real | REE_44 | n_detected | 13 | 13 | 0 | same |
| real | REE_44 | ne_at_bound | 1 | 1 | 0 | same |
| real | argon_noAr | r_squared | 0.2627 | 0.2569 | -0.005816 | worse |
| real | argon_noAr | median_stage_disagreement | 0.4241 | 0.3875 | -0.03665 | better |
| real | argon_noAr | n_detected | 9 | 9 | 0 | same |
| real | argon_noAr | ne_at_bound | 1 | 1 | 0 | same |
| real | scan9x9 | r_squared | 0.8305 | 0.8407 | 0.01013 | better |
| real | scan9x9 | median_stage_disagreement | 0.6732 | 0.6485 | -0.02466 | better |
| real | scan9x9 | n_detected | 17 | 16 | -1 | changed |
| real | scan9x9 | ne_at_bound | 1 | 1 | 0 | same |
| real | user_spec1 | r_squared | 0.9235 | 0.9192 | -0.004352 | worse |
| real | user_spec1 | median_stage_disagreement | 0.501 | 0.504 | 0.002951 | worse |
| real | user_spec1 | n_detected | 10 | 10 | 0 | same |
| real | user_spec1 | ne_at_bound | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | fe_fraction | 0.9111 | 0.8596 | -0.05145 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | non_fe_mass | 0.08891 | 0.1404 | 0.05145 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | abs_T_dev_10kK | 1.145 | 1.049 | -0.09654 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | fe_fraction | 0.01915 | 0.9156 | 0.8964 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | non_fe_mass | 0.9808 | 0.0844 | -0.8964 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | n_non_fe | 2 | 1 | -1 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | abs_T_dev_10kK | 0.07936 | 1.076 | 0.9971 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | fe_fraction | 0.9643 | 0.9636 | -0.0006988 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | non_fe_mass | 0.03567 | 0.03637 | 0.0006988 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | abs_T_dev_10kK | 1.231 | 1.231 | 0.0001834 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | fe_fraction | 0.9187 | 0.8701 | -0.04857 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | non_fe_mass | 0.08133 | 0.1299 | 0.04857 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | n_non_fe | 1 | 1 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | abs_T_dev_10kK | 1.047 | 1.045 | -0.001878 | better |
| fe | run-5291e643082343619a4973a1b67898db | fe_fraction | 0.07293 | 0.07936 | 0.00643 | better |
| fe | run-5291e643082343619a4973a1b67898db | non_fe_mass | 0.9271 | 0.9206 | -0.00643 | better |
| fe | run-5291e643082343619a4973a1b67898db | n_non_fe | 2 | 2 | 0 | same |
| fe | run-5291e643082343619a4973a1b67898db | abs_T_dev_10kK | 1.294 | 1.391 | 0.09634 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | fe_fraction | 0.9711 | 0.9501 | -0.02096 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | non_fe_mass | 0.02891 | 0.04987 | 0.02096 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | n_non_fe | 1 | 1 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | abs_T_dev_10kK | 1.32 | 1.319 | -0.001875 | better |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | fe_fraction | 0.105 | 0.1033 | -0.001699 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | non_fe_mass | 0.895 | 0.8967 | 0.001699 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | abs_T_dev_10kK | 1.391 | 1.379 | -0.01193 | better |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | fe_fraction | 0.9721 | 0.9685 | -0.003536 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | non_fe_mass | 0.02794 | 0.03148 | 0.003536 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | abs_T_dev_10kK | 1.437 | 1.396 | -0.0406 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | fe_fraction | 0.8652 | 0.8547 | -0.01051 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | non_fe_mass | 0.1348 | 0.1453 | 0.01051 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | abs_T_dev_10kK | 1.092 | 1.084 | -0.007796 | better |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | fe_fraction | 0.05806 | 0.0289 | -0.02916 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | non_fe_mass | 0.9419 | 0.9711 | 0.02916 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | abs_T_dev_10kK | 1.248 | 1.27 | 0.02166 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | fe_fraction | 0.007641 | 0.007778 | 0.0001371 | better |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | non_fe_mass | 0.9924 | 0.9922 | -0.0001371 | better |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | abs_T_dev_10kK | 1.146 | 1.145 | -0.001251 | better |
| fe | run-39998be9f14c479e8008efecec3bee25 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | fe_fraction | 0.06471 | 0.06456 | -0.0001439 | worse |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | non_fe_mass | 0.9353 | 0.9354 | 0.0001439 | worse |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | fe_fraction | 0.9922 | 0.9233 | -0.06897 | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | non_fe_mass | 0.007764 | 0.07673 | 0.06897 | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | abs_T_dev_10kK | 1.5 | 1.237 | -0.263 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | fe_fraction | 0.9541 | 0.882 | -0.07204 | worse |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | non_fe_mass | 0.04592 | 0.118 | 0.07204 | worse |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | abs_T_dev_10kK | 1.349 | 1.151 | -0.1981 | better |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | fe_fraction | 0.9246 | 0.9241 | -0.0004996 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | non_fe_mass | 0.07538 | 0.07588 | 0.0004996 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | abs_T_dev_10kK | 1.244 | 1.246 | 0.001703 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | fe_fraction | 0.9242 | 0.9428 | 0.01854 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | non_fe_mass | 0.07577 | 0.05722 | -0.01854 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | abs_T_dev_10kK | 1.233 | 1.231 | -0.002271 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | fe_fraction | 0 | 0.2574 | 0.2574 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | non_fe_mass | 1 | 0.7426 | -0.2574 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | n_non_fe | 2 | 1 | -1 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | abs_T_dev_10kK | 1.5 | 1.48 | -0.02041 | better |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | fe_fraction | 0.05329 | 0.07538 | 0.02209 | better |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | non_fe_mass | 0.9467 | 0.9246 | -0.02209 | better |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | n_non_fe | 1 | 1 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | abs_T_dev_10kK | 1.5 | 1.339 | -0.1607 | better |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | fe_fraction | 0.05444 | 0.05444 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | non_fe_mass | 0.9456 | 0.9456 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | fe_fraction | 0.7915 | 0.7915 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | non_fe_mass | 0.2085 | 0.2085 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | n_non_fe | 2 | 2 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | abs_T_dev_10kK | 1.304 | 1.304 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | fe_fraction | 0.6494 | 0.6488 | -0.0006288 | worse |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | non_fe_mass | 0.3506 | 0.3512 | 0.0006288 | worse |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | n_non_fe | 2 | 2 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | abs_T_dev_10kK | 1.168 | 1.171 | 0.002625 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | fe_fraction | 0.9947 | 0.9936 | -0.001112 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | non_fe_mass | 0.005264 | 0.006376 | 0.001112 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | abs_T_dev_10kK | 1.437 | 1.413 | -0.02345 | better |
| fe | run-90143c6b30884618ade5aea2f14819b8 | fe_fraction | 0.9927 | 0.9887 | -0.004 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | non_fe_mass | 0.007342 | 0.01134 | 0.004 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | abs_T_dev_10kK | 1.437 | 1.401 | -0.03558 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | fe_fraction | 0.9257 | 0.9014 | -0.0243 | worse |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | non_fe_mass | 0.07427 | 0.09857 | 0.0243 | worse |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | n_non_fe | 1 | 1 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | abs_T_dev_10kK | 1.187 | 1.139 | -0.04746 | better |

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 2 | 3 | 0 | 0 | 0.1559 | 0.1645 |
| synthetic | spurious_mass | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | abs_T_rel_err | 3 | 2 | 0 | 0 | 0.06862 | 0.07151 |
| synthetic | n_majors_missed | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | dominant_correct | 0 | 0 | 5 | 0 | 1 | 1 |
| real | r_squared | 3 | 2 | 0 | 0 | 0.4782 | 0.4803 |
| real | median_stage_disagreement | 3 | 2 | 0 | 0 | 0.4997 | 0.4985 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.8 | 11.6 |
| real | ne_at_bound | 1 | 0 | 4 | 0 | 1 | 0.8 |
| fe | fe_fraction | 6 | 16 | 4 | 0 | 0.5456 | 0.5788 |
| fe | non_fe_mass | 6 | 16 | 4 | 0 | 0.4544 | 0.4212 |
| fe | n_non_fe | 2 | 0 | 24 | 0 | 1.154 | 1.077 |
| fe | abs_T_dev_10kK | 15 | 6 | 5 | 0 | 1.276 | 1.283 |

## Verdict

- **synthetic**: no net change (5 better vs 5 worse metric-comparisons)
- **real**: improved (7 better vs 4 worse metric-comparisons)
- **fe**: regressed (29 better vs 38 worse metric-comparisons)
