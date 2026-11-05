# Fit-quality comparison: baseline -> sparse

baseline config: `{}`  
variant config: `{"sparsity": "adaptive-lasso"}`

## Per-item

| panel | item | metric | before | after | delta | verdict |
|---|---|---|---|---|---|---|
| synthetic | binary_ca_mg | median_rel_err | 0.1701 | 0.1701 | 0 | same |
| synthetic | binary_ca_mg | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | abs_T_rel_err | 0.1092 | 0.1092 | 0 | same |
| synthetic | binary_ca_mg | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | dominant_correct | 1 | 1 | 0 | same |
| synthetic | silicate_4el | median_rel_err | 0.5262 | 0.5075 | -0.01874 | better |
| synthetic | silicate_4el | spurious_mass | 0 | 0 | 0 | same |
| synthetic | silicate_4el | abs_T_rel_err | 0.09571 | 0.09441 | -0.001298 | better |
| synthetic | silicate_4el | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | silicate_4el | dominant_correct | 1 | 1 | 0 | same |
| synthetic | feldspar_k_na | median_rel_err | 0.1781 | 0.1781 | 0 | same |
| synthetic | feldspar_k_na | spurious_mass | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | abs_T_rel_err | 0.02691 | 0.02691 | 0 | same |
| synthetic | feldspar_k_na | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | dominant_correct | 1 | 1 | 0 | same |
| synthetic | iron_rich_low_t | status | error | error | - | n/a |
| synthetic | binary_argon_ambient | median_rel_err | 0.2173 | 0.2173 | 0 | same |
| synthetic | binary_argon_ambient | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | abs_T_rel_err | 0.1 | 0.1 | 0 | same |
| synthetic | binary_argon_ambient | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | dominant_correct | 0 | 0 | 0 | same |
| real | REE_01 | r_squared | -0.2939 | -0.2939 | 0 | same |
| real | REE_01 | median_stage_disagreement | 0.4306 | 0.4306 | 0 | same |
| real | REE_01 | n_detected | 10 | 10 | 0 | same |
| real | REE_01 | ne_at_bound | 1 | 1 | 0 | same |
| real | REE_44 | r_squared | 0.7208 | 0.7208 | -7.772e-16 | same |
| real | REE_44 | median_stage_disagreement | 0.4279 | 0.4279 | 9.12e-14 | same |
| real | REE_44 | n_detected | 14 | 14 | 0 | same |
| real | REE_44 | ne_at_bound | 1 | 1 | 0 | same |
| real | argon_noAr | r_squared | 0.2518 | 0.2515 | -0.000301 | worse |
| real | argon_noAr | median_stage_disagreement | 0.3714 | 0.3784 | 0.006949 | worse |
| real | argon_noAr | n_detected | 9 | 9 | 0 | same |
| real | argon_noAr | ne_at_bound | 1 | 1 | 0 | same |
| real | scan9x9 | r_squared | 0.7807 | 0.7825 | 0.001835 | better |
| real | scan9x9 | median_stage_disagreement | 0.6755 | 0.6778 | 0.00232 | worse |
| real | scan9x9 | n_detected | 16 | 16 | 0 | same |
| real | scan9x9 | ne_at_bound | 1 | 1 | 0 | same |
| real | user_spec1 | r_squared | 0.8879 | 0.8879 | 0 | same |
| real | user_spec1 | median_stage_disagreement | 0.5082 | 0.5082 | 0 | same |
| real | user_spec1 | n_detected | 10 | 10 | 0 | same |
| real | user_spec1 | ne_at_bound | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | fe_fraction | 0.4138 | 0.4138 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | non_fe_mass | 0.5862 | 0.5862 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | abs_T_dev_10kK | 1.215 | 1.215 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | fe_fraction | 0.02125 | 0.02125 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | non_fe_mass | 0.9787 | 0.9787 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | abs_T_dev_10kK | 0.07936 | 0.07936 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | fe_fraction | 1 | 1 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | non_fe_mass | 0 | 0 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | abs_T_dev_10kK | 1.217 | 1.217 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | fe_fraction | 0.2624 | 0.2624 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | non_fe_mass | 0.7376 | 0.7376 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | n_non_fe | 1 | 1 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | abs_T_dev_10kK | 1.064 | 1.064 | 0 | same |
| fe | run-5291e643082343619a4973a1b67898db | fe_fraction | 0.3022 | 0.07641 | -0.2258 | worse |
| fe | run-5291e643082343619a4973a1b67898db | non_fe_mass | 0.6978 | 0.9236 | 0.2258 | worse |
| fe | run-5291e643082343619a4973a1b67898db | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-5291e643082343619a4973a1b67898db | abs_T_dev_10kK | 1.5 | 1.391 | -0.1092 | better |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | fe_fraction | 1 | 1 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | non_fe_mass | 0 | 0 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | n_non_fe | 0 | 0 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | abs_T_dev_10kK | 1.42 | 1.42 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | fe_fraction | 0.07643 | 0.07643 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | non_fe_mass | 0.9236 | 0.9236 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | abs_T_dev_10kK | 1.391 | 1.391 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | fe_fraction | 1 | 1 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | non_fe_mass | 0 | 0 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | abs_T_dev_10kK | 1.437 | 1.437 | 0.0001363 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | fe_fraction | 0.2858 | 0.2858 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | non_fe_mass | 0.7142 | 0.7142 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | abs_T_dev_10kK | 1.093 | 1.093 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | fe_fraction | 0.0126 | 0.009344 | -0.003259 | worse |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | non_fe_mass | 0.9874 | 0.9907 | 0.003259 | worse |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | abs_T_dev_10kK | 1.437 | 1.5 | 0.06346 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | fe_fraction | 0.04748 | 0.04748 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | non_fe_mass | 0.9525 | 0.9525 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | n_non_fe | 3 | 3 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | abs_T_dev_10kK | 1.03 | 1.03 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | fe_fraction | 0.07838 | 0.07835 | -2.771e-05 | worse |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | non_fe_mass | 0.9216 | 0.9216 | 2.771e-05 | worse |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | fe_fraction | 1 | 1 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | non_fe_mass | 0 | 0 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | abs_T_dev_10kK | 1.437 | 1.5 | 0.06348 | worse |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | fe_fraction | 0.3222 | 0.3224 | 0.0002571 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | non_fe_mass | 0.6778 | 0.6776 | -0.0002571 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | abs_T_dev_10kK | 1.236 | 1.237 | 0.0005908 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | fe_fraction | 1 | 1 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | non_fe_mass | 0 | 0 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | abs_T_dev_10kK | 1.421 | 1.421 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | fe_fraction | 0.3409 | 0.5562 | 0.2153 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | non_fe_mass | 0.6591 | 0.4438 | -0.2153 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | abs_T_dev_10kK | 1.23 | 1.227 | -0.003156 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | fe_fraction | 0.02626 | 0.02626 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | non_fe_mass | 0.9737 | 0.9737 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | n_non_fe | 3 | 3 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | fe_fraction | 0.06125 | 0.06179 | 0.0005403 | better |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | non_fe_mass | 0.9388 | 0.9382 | -0.0005403 | better |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | n_non_fe | 2 | 2 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | fe_fraction | 0.04406 | 0.04406 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | non_fe_mass | 0.9559 | 0.9559 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | fe_fraction | 1 | 1 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | non_fe_mass | 0 | 0 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | n_non_fe | 0 | 0 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | abs_T_dev_10kK | 1.278 | 1.278 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | fe_fraction | 0.4001 | 0.4001 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | non_fe_mass | 0.5999 | 0.5999 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | n_non_fe | 1 | 1 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | abs_T_dev_10kK | 1.309 | 1.309 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | fe_fraction | 0.8398 | 0.5767 | -0.2631 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | non_fe_mass | 0.1602 | 0.4233 | 0.2631 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | abs_T_dev_10kK | 1.437 | 1.437 | 0.0002295 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | fe_fraction | 0.7933 | 0.7665 | -0.02686 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | non_fe_mass | 0.2067 | 0.2335 | 0.02686 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | fe_fraction | 0.5631 | 0.5631 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | non_fe_mass | 0.4369 | 0.4369 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | n_non_fe | 1 | 1 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 1 | 0 | 3 | 0 | 0.2729 | 0.2682 |
| synthetic | spurious_mass | 0 | 0 | 4 | 0 | 0 | 0 |
| synthetic | abs_T_rel_err | 1 | 0 | 3 | 0 | 0.08296 | 0.08264 |
| synthetic | n_majors_missed | 0 | 0 | 4 | 0 | 0 | 0 |
| synthetic | dominant_correct | 0 | 0 | 4 | 0 | 0.75 | 0.75 |
| real | r_squared | 1 | 1 | 3 | 0 | 0.4695 | 0.4698 |
| real | median_stage_disagreement | 0 | 2 | 3 | 0 | 0.4827 | 0.4846 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.8 | 11.8 |
| real | ne_at_bound | 0 | 0 | 5 | 0 | 1 | 1 |
| fe | fe_fraction | 3 | 5 | 18 | 0 | 0.4189 | 0.4073 |
| fe | non_fe_mass | 3 | 5 | 18 | 0 | 0.5811 | 0.5927 |
| fe | n_non_fe | 0 | 1 | 25 | 0 | 1.154 | 1.192 |
| fe | abs_T_dev_10kK | 2 | 5 | 19 | 0 | 1.307 | 1.307 |

## Verdict

- **synthetic**: improved (2 better vs 0 worse metric-comparisons)
- **real**: regressed (1 better vs 3 worse metric-comparisons)
- **fe**: regressed (8 better vs 16 worse metric-comparisons)
