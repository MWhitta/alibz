# Fit-quality comparison: w3t-base -> w3t-offset

baseline config: `{}`  
variant config: `{"stage_temperature_offset_max_k": 2000}`

## Per-item

| panel | item | metric | before | after | delta | verdict |
|---|---|---|---|---|---|---|
| synthetic | binary_ca_mg | median_rel_err | 0.1666 | 0.1787 | 0.01216 | worse |
| synthetic | binary_ca_mg | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | abs_T_rel_err | 0.1117 | 0.2305 | 0.1188 | worse |
| synthetic | binary_ca_mg | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | dominant_correct | 1 | 1 | 0 | same |
| synthetic | silicate_4el | median_rel_err | 0.3615 | 0.1678 | -0.1937 | better |
| synthetic | silicate_4el | spurious_mass | 0 | 0.0008121 | 0.0008121 | worse |
| synthetic | silicate_4el | abs_T_rel_err | 0.1048 | 0.2176 | 0.1128 | worse |
| synthetic | silicate_4el | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | silicate_4el | dominant_correct | 1 | 1 | 0 | same |
| synthetic | feldspar_k_na | median_rel_err | 0.08967 | 0.08967 | 0 | same |
| synthetic | feldspar_k_na | spurious_mass | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | abs_T_rel_err | 0.01215 | 0.01215 | 0 | same |
| synthetic | feldspar_k_na | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | dominant_correct | 1 | 1 | 0 | same |
| synthetic | iron_rich_low_t | median_rel_err | 0.009964 | 0.0169 | 0.006936 | worse |
| synthetic | iron_rich_low_t | spurious_mass | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | abs_T_rel_err | 0.001947 | 0.000312 | -0.001635 | better |
| synthetic | iron_rich_low_t | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | dominant_correct | 1 | 1 | 0 | same |
| synthetic | binary_argon_ambient | median_rel_err | 0.152 | 0.1352 | -0.01676 | better |
| synthetic | binary_argon_ambient | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | abs_T_rel_err | 0.1125 | 0.2305 | 0.1179 | worse |
| synthetic | binary_argon_ambient | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | dominant_correct | 1 | 1 | 0 | same |
| real | REE_01 | r_squared | -0.2883 | -0.3133 | -0.02502 | worse |
| real | REE_01 | median_stage_disagreement | 0.4263 | 0.9874 | 0.5611 | worse |
| real | REE_01 | n_detected | 10 | 10 | 0 | same |
| real | REE_01 | ne_at_bound | 1 | 1 | 0 | same |
| real | REE_44 | r_squared | 0.6625 | 0.6622 | -0.0002568 | worse |
| real | REE_44 | median_stage_disagreement | 0.4737 | 0.6038 | 0.1301 | worse |
| real | REE_44 | n_detected | 13 | 13 | 0 | same |
| real | REE_44 | ne_at_bound | 1 | 1 | 0 | same |
| real | argon_noAr | r_squared | 0.2627 | 0.3244 | 0.06166 | better |
| real | argon_noAr | median_stage_disagreement | 0.4241 | 0.4333 | 0.009151 | worse |
| real | argon_noAr | n_detected | 9 | 9 | 0 | same |
| real | argon_noAr | ne_at_bound | 1 | 1 | 0 | same |
| real | scan9x9 | r_squared | 0.8305 | 0.8503 | 0.01975 | better |
| real | scan9x9 | median_stage_disagreement | 0.6732 | 0.8086 | 0.1354 | worse |
| real | scan9x9 | n_detected | 17 | 16 | -1 | changed |
| real | scan9x9 | ne_at_bound | 1 | 1 | 0 | same |
| real | user_spec1 | r_squared | 0.9235 | 0.9198 | -0.003737 | worse |
| real | user_spec1 | median_stage_disagreement | 0.501 | 0.5911 | 0.09007 | worse |
| real | user_spec1 | n_detected | 10 | 10 | 0 | same |
| real | user_spec1 | ne_at_bound | 1 | 0 | -1 | better |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | fe_fraction | 0.9111 | 0.9088 | -0.002313 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | non_fe_mass | 0.08891 | 0.09122 | 0.002313 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | abs_T_dev_10kK | 1.145 | 1.141 | -0.00452 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | fe_fraction | 0.01915 | 0.01843 | -0.0007224 | worse |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | non_fe_mass | 0.9808 | 0.9816 | 0.0007224 | worse |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | abs_T_dev_10kK | 0.07936 | 0.0793 | -6.348e-05 | better |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | fe_fraction | 0.9643 | 0.9643 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | non_fe_mass | 0.03567 | 0.03567 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | abs_T_dev_10kK | 1.231 | 1.231 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | fe_fraction | 0.9187 | 0.9187 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | non_fe_mass | 0.08133 | 0.08133 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | n_non_fe | 1 | 1 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | abs_T_dev_10kK | 1.047 | 1.047 | 0 | same |
| fe | run-5291e643082343619a4973a1b67898db | fe_fraction | 0.07293 | 0.07293 | 0 | same |
| fe | run-5291e643082343619a4973a1b67898db | non_fe_mass | 0.9271 | 0.9271 | 0 | same |
| fe | run-5291e643082343619a4973a1b67898db | n_non_fe | 2 | 2 | 0 | same |
| fe | run-5291e643082343619a4973a1b67898db | abs_T_dev_10kK | 1.294 | 1.294 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | fe_fraction | 0.9711 | 0.9711 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | non_fe_mass | 0.02891 | 0.02891 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | n_non_fe | 1 | 1 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | abs_T_dev_10kK | 1.32 | 1.32 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | fe_fraction | 0.105 | 0.105 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | non_fe_mass | 0.895 | 0.895 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | abs_T_dev_10kK | 1.391 | 1.391 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | fe_fraction | 0.9721 | 0.9721 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | non_fe_mass | 0.02794 | 0.02794 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | fe_fraction | 0.8652 | 0.8652 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | non_fe_mass | 0.1348 | 0.1348 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | abs_T_dev_10kK | 1.092 | 1.092 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | fe_fraction | 0.05806 | 0.0532 | -0.004867 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | non_fe_mass | 0.9419 | 0.9468 | 0.004867 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | abs_T_dev_10kK | 1.248 | 1.268 | 0.02039 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | fe_fraction | 0.007641 | 0.008243 | 0.0006023 | better |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | non_fe_mass | 0.9924 | 0.9918 | -0.0006023 | better |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | abs_T_dev_10kK | 1.146 | 1.144 | -0.002492 | better |
| fe | run-39998be9f14c479e8008efecec3bee25 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | fe_fraction | 0.06471 | 0.06296 | -0.001745 | worse |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | non_fe_mass | 0.9353 | 0.937 | 0.001745 | worse |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | abs_T_dev_10kK | 1.437 | 1.391 | -0.0461 | better |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | fe_fraction | 0.9922 | 0.9922 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | non_fe_mass | 0.007764 | 0.007764 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | fe_fraction | 0.9541 | 0.9541 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | non_fe_mass | 0.04592 | 0.04592 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | abs_T_dev_10kK | 1.349 | 1.349 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | fe_fraction | 0.9246 | 0.924 | -0.0006085 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | non_fe_mass | 0.07538 | 0.07599 | 0.0006085 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | abs_T_dev_10kK | 1.244 | 1.246 | 0.002033 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | fe_fraction | 0.9242 | 0.9242 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | non_fe_mass | 0.07577 | 0.07577 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | abs_T_dev_10kK | 1.233 | 1.233 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | fe_fraction | 0 | 0 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | fe_fraction | 0.05329 | 0.05329 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | non_fe_mass | 0.9467 | 0.9467 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | n_non_fe | 1 | 1 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
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
| fe | run-01105eaef0044dca87fb08bfdea5b107 | fe_fraction | 0.9947 | 0.9947 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | non_fe_mass | 0.005264 | 0.005264 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | abs_T_dev_10kK | 1.437 | 1.437 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | fe_fraction | 0.9927 | 0.4215 | -0.5711 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | non_fe_mass | 0.007342 | 0.5785 | 0.5711 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | abs_T_dev_10kK | 1.437 | 0.2562 | -1.181 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | fe_fraction | 0.9257 | 0.9251 | -0.0006749 | worse |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | non_fe_mass | 0.07427 | 0.07495 | 0.0006749 | worse |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | n_non_fe | 1 | 1 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | abs_T_dev_10kK | 1.187 | 1.187 | 2.686e-05 | worse |

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 2 | 2 | 1 | 0 | 0.1559 | 0.1177 |
| synthetic | spurious_mass | 0 | 1 | 4 | 0 | 0 | 0.0001624 |
| synthetic | abs_T_rel_err | 1 | 3 | 1 | 0 | 0.06862 | 0.1382 |
| synthetic | n_majors_missed | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | dominant_correct | 0 | 0 | 5 | 0 | 1 | 1 |
| real | r_squared | 2 | 3 | 0 | 0 | 0.4782 | 0.4887 |
| real | median_stage_disagreement | 0 | 5 | 0 | 0 | 0.4997 | 0.6848 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.8 | 11.6 |
| real | ne_at_bound | 1 | 0 | 4 | 0 | 1 | 0.8 |
| fe | fe_fraction | 1 | 8 | 17 | 0 | 0.5456 | 0.5233 |
| fe | non_fe_mass | 1 | 8 | 17 | 0 | 0.4544 | 0.4767 |
| fe | n_non_fe | 0 | 0 | 26 | 0 | 1.154 | 1.154 |
| fe | abs_T_dev_10kK | 5 | 4 | 17 | 0 | 1.276 | 1.229 |

## Verdict

- **synthetic**: regressed (3 better vs 6 worse metric-comparisons)
- **real**: regressed (3 better vs 8 worse metric-comparisons)
- **fe**: regressed (7 better vs 20 worse metric-comparisons)
