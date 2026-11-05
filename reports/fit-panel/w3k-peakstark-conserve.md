# Fit-quality comparison: w3k-base -> w3k-peakstark-conserve

baseline config: `{}`  
variant config: `{"overlap_kernel": "peak+stark"}`

## Per-item

| panel | item | metric | before | after | delta | verdict |
|---|---|---|---|---|---|---|
| synthetic | binary_ca_mg | median_rel_err | 0.1666 | 0.09618 | -0.07039 | better |
| synthetic | binary_ca_mg | spurious_mass | 0 | 0.0004251 | 0.0004251 | worse |
| synthetic | binary_ca_mg | abs_T_rel_err | 0.1117 | 0.08644 | -0.02522 | better |
| synthetic | binary_ca_mg | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | dominant_correct | 1 | 1 | 0 | same |
| synthetic | silicate_4el | median_rel_err | 0.3615 | 0.1807 | -0.1808 | better |
| synthetic | silicate_4el | spurious_mass | 0 | 0 | 0 | same |
| synthetic | silicate_4el | abs_T_rel_err | 0.1048 | 0.0774 | -0.02745 | better |
| synthetic | silicate_4el | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | silicate_4el | dominant_correct | 1 | 1 | 0 | same |
| synthetic | feldspar_k_na | median_rel_err | 0.08967 | 0.1566 | 0.06697 | worse |
| synthetic | feldspar_k_na | spurious_mass | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | abs_T_rel_err | 0.01215 | 0.1019 | 0.08971 | worse |
| synthetic | feldspar_k_na | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | dominant_correct | 1 | 1 | 0 | same |
| synthetic | iron_rich_low_t | median_rel_err | 0.009964 | 0.09297 | 0.083 | worse |
| synthetic | iron_rich_low_t | spurious_mass | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | abs_T_rel_err | 0.001947 | 0.01116 | 0.009214 | worse |
| synthetic | iron_rich_low_t | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | dominant_correct | 1 | 1 | 0 | same |
| synthetic | binary_argon_ambient | median_rel_err | 0.152 | 0.1102 | -0.04177 | better |
| synthetic | binary_argon_ambient | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | abs_T_rel_err | 0.1125 | 0.08773 | -0.02479 | better |
| synthetic | binary_argon_ambient | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | dominant_correct | 1 | 1 | 0 | same |
| real | REE_01 | r_squared | -0.2883 | -0.2781 | 0.01015 | better |
| real | REE_01 | median_stage_disagreement | 0.4263 | 0.5401 | 0.1138 | worse |
| real | REE_01 | n_detected | 10 | 10 | 0 | same |
| real | REE_01 | ne_at_bound | 1 | 1 | 0 | same |
| real | REE_44 | r_squared | 0.6625 | 0.6433 | -0.0192 | worse |
| real | REE_44 | median_stage_disagreement | 0.4737 | 0.3092 | -0.1645 | better |
| real | REE_44 | n_detected | 13 | 13 | 0 | same |
| real | REE_44 | ne_at_bound | 1 | 1 | 0 | same |
| real | argon_noAr | r_squared | 0.2627 | 0.2995 | 0.03679 | better |
| real | argon_noAr | median_stage_disagreement | 0.4241 | 0.3177 | -0.1065 | better |
| real | argon_noAr | n_detected | 9 | 9 | 0 | same |
| real | argon_noAr | ne_at_bound | 1 | 1 | 0 | same |
| real | scan9x9 | r_squared | 0.8305 | 0.8306 | 8.255e-05 | better |
| real | scan9x9 | median_stage_disagreement | 0.6732 | 0.4557 | -0.2174 | better |
| real | scan9x9 | n_detected | 17 | 16 | -1 | changed |
| real | scan9x9 | ne_at_bound | 1 | 1 | 0 | same |
| real | user_spec1 | r_squared | 0.9235 | 0.9158 | -0.007749 | worse |
| real | user_spec1 | median_stage_disagreement | 0.501 | 0.6005 | 0.09944 | worse |
| real | user_spec1 | n_detected | 10 | 10 | 0 | same |
| real | user_spec1 | ne_at_bound | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | fe_fraction | 0.9111 | 0.964 | 0.05296 | better |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | non_fe_mass | 0.08891 | 0.03595 | -0.05296 | better |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | abs_T_dev_10kK | 1.145 | 1.279 | 0.1336 | worse |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | fe_fraction | 0.01915 | 0.04057 | 0.02142 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | non_fe_mass | 0.9808 | 0.9594 | -0.02142 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | n_non_fe | 2 | 1 | -1 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | abs_T_dev_10kK | 0.07936 | 0 | -0.07936 | better |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | fe_fraction | 0.9643 | 0.9493 | -0.01503 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | non_fe_mass | 0.03567 | 0.0507 | 0.01503 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | abs_T_dev_10kK | 1.231 | 1.368 | 0.1368 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | fe_fraction | 0.9187 | 1 | 0.08133 | better |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | non_fe_mass | 0.08133 | 0 | -0.08133 | better |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | n_non_fe | 1 | 0 | -1 | better |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | abs_T_dev_10kK | 1.047 | 1.437 | 0.3898 | worse |
| fe | run-5291e643082343619a4973a1b67898db | fe_fraction | 0.07293 | 0.07576 | 0.002831 | better |
| fe | run-5291e643082343619a4973a1b67898db | non_fe_mass | 0.9271 | 0.9242 | -0.002831 | better |
| fe | run-5291e643082343619a4973a1b67898db | n_non_fe | 2 | 2 | 0 | same |
| fe | run-5291e643082343619a4973a1b67898db | abs_T_dev_10kK | 1.294 | 1.391 | 0.09665 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | fe_fraction | 0.9711 | 0.9626 | -0.008489 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | non_fe_mass | 0.02891 | 0.0374 | 0.008489 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | n_non_fe | 1 | 1 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | abs_T_dev_10kK | 1.32 | 1.391 | 0.07026 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | fe_fraction | 0.105 | 0.08086 | -0.02412 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | non_fe_mass | 0.895 | 0.9191 | 0.02412 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | abs_T_dev_10kK | 1.391 | 1.391 | 2.724e-05 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | fe_fraction | 0.9721 | 0.9702 | -0.001906 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | non_fe_mass | 0.02794 | 0.02985 | 0.001906 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | abs_T_dev_10kK | 1.437 | 1.391 | -0.04598 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | fe_fraction | 0.8652 | 0.9413 | 0.07608 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | non_fe_mass | 0.1348 | 0.05868 | -0.07608 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | abs_T_dev_10kK | 1.092 | 1.227 | 0.1354 | worse |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | fe_fraction | 0.05806 | 0.03203 | -0.02603 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | non_fe_mass | 0.9419 | 0.968 | 0.02603 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | abs_T_dev_10kK | 1.248 | 1.263 | 0.01559 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | fe_fraction | 0.007641 | 0.007611 | -2.998e-05 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | non_fe_mass | 0.9924 | 0.9924 | 2.998e-05 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | abs_T_dev_10kK | 1.146 | 1.094 | -0.05188 | better |
| fe | run-39998be9f14c479e8008efecec3bee25 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | fe_fraction | 0.06471 | 0.07511 | 0.0104 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | non_fe_mass | 0.9353 | 0.9249 | -0.0104 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | abs_T_dev_10kK | 1.437 | 1.389 | -0.04792 | better |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | fe_fraction | 0.9922 | 0.9749 | -0.01737 | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | non_fe_mass | 0.007764 | 0.02514 | 0.01737 | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | abs_T_dev_10kK | 1.5 | 1.437 | -0.06348 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | fe_fraction | 0.9541 | 0.9723 | 0.01823 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | non_fe_mass | 0.04592 | 0.02769 | -0.01823 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | abs_T_dev_10kK | 1.349 | 1.432 | 0.08256 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | fe_fraction | 0.9246 | 0.933 | 0.008361 | better |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | non_fe_mass | 0.07538 | 0.06702 | -0.008361 | better |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | abs_T_dev_10kK | 1.244 | 1.217 | -0.02726 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | fe_fraction | 0.9242 | 0.955 | 0.03076 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | non_fe_mass | 0.07577 | 0.04501 | -0.03076 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | abs_T_dev_10kK | 1.233 | 1.329 | 0.09589 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | fe_fraction | 0 | 0 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | fe_fraction | 0.05329 | 0.04658 | -0.006712 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | non_fe_mass | 0.9467 | 0.9534 | 0.006712 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | n_non_fe | 1 | 1 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | fe_fraction | 0.05444 | 0.05221 | -0.002229 | worse |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | non_fe_mass | 0.9456 | 0.9478 | 0.002229 | worse |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | abs_T_dev_10kK | 1.437 | 1.435 | -0.001531 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | fe_fraction | 0.7915 | 1 | 0.2085 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | non_fe_mass | 0.2085 | 0 | -0.2085 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | n_non_fe | 2 | 0 | -2 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | abs_T_dev_10kK | 1.304 | 1.247 | -0.05639 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | fe_fraction | 0.6494 | 0.6995 | 0.05009 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | non_fe_mass | 0.3506 | 0.3005 | -0.05009 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | n_non_fe | 2 | 2 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | abs_T_dev_10kK | 1.168 | 1.171 | 0.002625 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | fe_fraction | 0.9947 | 0.9808 | -0.01393 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | non_fe_mass | 0.005264 | 0.01919 | 0.01393 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | abs_T_dev_10kK | 1.437 | 1.391 | -0.04613 | better |
| fe | run-90143c6b30884618ade5aea2f14819b8 | fe_fraction | 0.9927 | 0.986 | -0.006619 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | non_fe_mass | 0.007342 | 0.01396 | 0.006619 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | abs_T_dev_10kK | 1.437 | 1.437 | -9.323e-05 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | fe_fraction | 0.9257 | 0.9351 | 0.009354 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | non_fe_mass | 0.07427 | 0.06492 | -0.009354 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | n_non_fe | 1 | 1 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | abs_T_dev_10kK | 1.187 | 1.193 | 0.006789 | worse |

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 3 | 2 | 0 | 0 | 0.1559 | 0.1273 |
| synthetic | spurious_mass | 0 | 1 | 4 | 0 | 0 | 8.502e-05 |
| synthetic | abs_T_rel_err | 3 | 2 | 0 | 0 | 0.06862 | 0.07292 |
| synthetic | n_majors_missed | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | dominant_correct | 0 | 0 | 5 | 0 | 1 | 1 |
| real | r_squared | 3 | 2 | 0 | 0 | 0.4782 | 0.4822 |
| real | median_stage_disagreement | 3 | 2 | 0 | 0 | 0.4997 | 0.4446 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.8 | 11.6 |
| real | ne_at_bound | 0 | 0 | 5 | 0 | 1 | 1 |
| fe | fe_fraction | 12 | 11 | 3 | 0 | 0.5456 | 0.5629 |
| fe | non_fe_mass | 12 | 11 | 3 | 0 | 0.4544 | 0.4371 |
| fe | n_non_fe | 3 | 0 | 23 | 0 | 1.154 | 1 |
| fe | abs_T_dev_10kK | 10 | 12 | 4 | 0 | 1.276 | 1.304 |

## Verdict

- **synthetic**: improved (6 better vs 5 worse metric-comparisons)
- **real**: improved (6 better vs 4 worse metric-comparisons)
- **fe**: improved (37 better vs 34 worse metric-comparisons)
