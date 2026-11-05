# Fit-quality comparison: baseline -> huber

baseline config: `{}`  
variant config: `{"robust_loss": "huber"}`

## Per-item

| panel | item | metric | before | after | delta | verdict |
|---|---|---|---|---|---|---|
| synthetic | binary_ca_mg | median_rel_err | 0.1701 | 0.01122 | -0.1588 | better |
| synthetic | binary_ca_mg | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | abs_T_rel_err | 0.1092 | 0.09811 | -0.01108 | better |
| synthetic | binary_ca_mg | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | dominant_correct | 1 | 1 | 0 | same |
| synthetic | silicate_4el | median_rel_err | 0.5262 | 0.8531 | 0.327 | worse |
| synthetic | silicate_4el | spurious_mass | 0 | 0 | 0 | same |
| synthetic | silicate_4el | abs_T_rel_err | 0.09571 | 0.8727 | 0.777 | worse |
| synthetic | silicate_4el | n_majors_missed | 0 | 1 | 1 | worse |
| synthetic | silicate_4el | dominant_correct | 1 | 1 | 0 | same |
| synthetic | feldspar_k_na | median_rel_err | 0.1781 | 0.9201 | 0.742 | worse |
| synthetic | feldspar_k_na | spurious_mass | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | abs_T_rel_err | 0.02691 | 0.3623 | 0.3354 | worse |
| synthetic | feldspar_k_na | n_majors_missed | 0 | 2 | 2 | worse |
| synthetic | feldspar_k_na | dominant_correct | 1 | 1 | 0 | same |
| synthetic | iron_rich_low_t | status | error | ok | - | better |
| synthetic | binary_argon_ambient | median_rel_err | 0.2173 | 0.1152 | -0.102 | better |
| synthetic | binary_argon_ambient | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | abs_T_rel_err | 0.1 | 0.1143 | 0.01426 | worse |
| synthetic | binary_argon_ambient | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | dominant_correct | 0 | 1 | 1 | better |
| real | REE_01 | r_squared | -0.2939 | -0.3426 | -0.04872 | worse |
| real | REE_01 | median_stage_disagreement | 0.4306 | 0.3927 | -0.03789 | better |
| real | REE_01 | n_detected | 10 | 10 | 0 | same |
| real | REE_01 | ne_at_bound | 1 | 1 | 0 | same |
| real | REE_44 | r_squared | 0.7208 | 0.6575 | -0.06335 | worse |
| real | REE_44 | median_stage_disagreement | 0.4279 | 0.4304 | 0.002463 | worse |
| real | REE_44 | n_detected | 14 | 13 | -1 | changed |
| real | REE_44 | ne_at_bound | 1 | 1 | 0 | same |
| real | argon_noAr | r_squared | 0.2518 | 0.2181 | -0.03368 | worse |
| real | argon_noAr | median_stage_disagreement | 0.3714 | 0.3648 | -0.006635 | better |
| real | argon_noAr | n_detected | 9 | 9 | 0 | same |
| real | argon_noAr | ne_at_bound | 1 | 1 | 0 | same |
| real | scan9x9 | r_squared | 0.7807 | 0.4291 | -0.3515 | worse |
| real | scan9x9 | median_stage_disagreement | 0.6755 | 0.6365 | -0.03894 | better |
| real | scan9x9 | n_detected | 16 | 16 | 0 | same |
| real | scan9x9 | ne_at_bound | 1 | 1 | 0 | same |
| real | user_spec1 | r_squared | 0.8879 | 0.843 | -0.04492 | worse |
| real | user_spec1 | median_stage_disagreement | 0.5082 | 0.9895 | 0.4813 | worse |
| real | user_spec1 | n_detected | 10 | 7 | -3 | changed |
| real | user_spec1 | ne_at_bound | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | fe_fraction | 0.4138 | 0.3851 | -0.02871 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | non_fe_mass | 0.5862 | 0.6149 | 0.02871 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | abs_T_dev_10kK | 1.215 | 1.155 | -0.06016 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | fe_fraction | 0.02125 | 0.02085 | -0.0003986 | worse |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | non_fe_mass | 0.9787 | 0.9791 | 0.0003986 | worse |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | abs_T_dev_10kK | 0.07936 | 0.07936 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | fe_fraction | 1 | 1 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | non_fe_mass | 0 | 0 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | abs_T_dev_10kK | 1.217 | 1.207 | -0.009596 | better |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | fe_fraction | 0.2624 | 0.2733 | 0.01087 | better |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | non_fe_mass | 0.7376 | 0.7267 | -0.01087 | better |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | n_non_fe | 1 | 1 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | abs_T_dev_10kK | 1.064 | 1.007 | -0.0569 | better |
| fe | run-5291e643082343619a4973a1b67898db | fe_fraction | 0.3022 | 0.05631 | -0.2459 | worse |
| fe | run-5291e643082343619a4973a1b67898db | non_fe_mass | 0.6978 | 0.9437 | 0.2459 | worse |
| fe | run-5291e643082343619a4973a1b67898db | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-5291e643082343619a4973a1b67898db | abs_T_dev_10kK | 1.5 | 1.391 | -0.1092 | better |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | status | ok | error | - | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | status | ok | error | - | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | status | ok | error | - | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | status | ok | error | - | worse |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | status | ok | error | - | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | status | ok | error | - | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | status | ok | error | - | worse |
| fe | run-39998be9f14c479e8008efecec3bee25 | status | ok | error | - | worse |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | status | ok | error | - | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | status | ok | error | - | worse |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | status | ok | error | - | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | fe_fraction | 1 | 0.463 | -0.537 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | non_fe_mass | 0 | 0.537 | 0.537 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | n_non_fe | 0 | 1 | 1 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | abs_T_dev_10kK | 1.421 | 1.437 | 0.01636 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | fe_fraction | 0.3409 | 0.3115 | -0.02935 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | non_fe_mass | 0.6591 | 0.6885 | 0.02935 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | abs_T_dev_10kK | 1.23 | 1.175 | -0.05529 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | fe_fraction | 0.02626 | 0.03088 | 0.004619 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | non_fe_mass | 0.9737 | 0.9691 | -0.004619 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | n_non_fe | 3 | 2 | -1 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | fe_fraction | 0.06125 | 0.05544 | -0.005806 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | non_fe_mass | 0.9388 | 0.9446 | 0.005806 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | n_non_fe | 2 | 2 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | fe_fraction | 0.04406 | 0.03814 | -0.00592 | worse |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | non_fe_mass | 0.9559 | 0.9619 | 0.00592 | worse |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | abs_T_dev_10kK | 1.437 | 1.358 | -0.07857 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | fe_fraction | 1 | 1 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | non_fe_mass | 0 | 0 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | n_non_fe | 0 | 0 | 0 | same |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | abs_T_dev_10kK | 1.278 | 1.255 | -0.02313 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | fe_fraction | 0.4001 | 0.392 | -0.008102 | worse |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | non_fe_mass | 0.5999 | 0.608 | 0.008102 | worse |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | n_non_fe | 1 | 1 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | abs_T_dev_10kK | 1.309 | 1.279 | -0.03022 | better |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | fe_fraction | 0.8398 | 0.5775 | -0.2623 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | non_fe_mass | 0.1602 | 0.4225 | 0.2623 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | abs_T_dev_10kK | 1.437 | 1.437 | 0.0001434 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | fe_fraction | 0.7933 | 0.7969 | 0.003582 | better |
| fe | run-90143c6b30884618ade5aea2f14819b8 | non_fe_mass | 0.2067 | 0.2031 | -0.003582 | better |
| fe | run-90143c6b30884618ade5aea2f14819b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | abs_T_dev_10kK | 1.437 | 1.432 | -0.005181 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | fe_fraction | 0.5631 | 0.5655 | 0.002394 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | non_fe_mass | 0.4369 | 0.4345 | -0.002394 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | n_non_fe | 1 | 1 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 2 | 2 | 0 | 0 | 0.2729 | 0.3975 |
| synthetic | spurious_mass | 0 | 0 | 4 | 0 | 0 | 0 |
| synthetic | abs_T_rel_err | 1 | 3 | 0 | 0 | 0.08296 | 0.3395 |
| synthetic | n_majors_missed | 0 | 2 | 2 | 0 | 0 | 0.6 |
| synthetic | dominant_correct | 1 | 0 | 3 | 0 | 0.75 | 1 |
| real | r_squared | 0 | 5 | 0 | 0 | 0.4695 | 0.361 |
| real | median_stage_disagreement | 3 | 2 | 0 | 0 | 0.4827 | 0.5628 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.8 | 11 |
| real | ne_at_bound | 0 | 0 | 5 | 0 | 1 | 1 |
| fe | fe_fraction | 4 | 9 | 2 | 0 | 0.4189 | 0.3978 |
| fe | non_fe_mass | 4 | 9 | 2 | 0 | 0.5811 | 0.6022 |
| fe | n_non_fe | 1 | 2 | 12 | 0 | 1.154 | 1.267 |
| fe | abs_T_dev_10kK | 9 | 2 | 4 | 0 | 1.307 | 1.243 |

## Verdict

- **synthetic**: regressed (4 better vs 7 worse metric-comparisons)
- **real**: regressed (3 better vs 7 worse metric-comparisons)
- **fe**: regressed (18 better vs 22 worse metric-comparisons)
