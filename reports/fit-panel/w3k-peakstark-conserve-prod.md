# Fit-quality comparison: cpi -> w3k-peakstark-conserve-prod

baseline config: `{}`  
variant config: `{"overlap_kernel": "peak+stark"}`

## Per-item

| panel | item | metric | before | after | delta | verdict |
|---|---|---|---|---|---|---|
| synthetic | binary_ca_mg | median_rel_err | 0.2352 | 0.151 | -0.08424 | better |
| synthetic | binary_ca_mg | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | abs_T_rel_err | 0.1116 | 0.07394 | -0.03761 | better |
| synthetic | binary_ca_mg | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | dominant_correct | 0 | 1 | 1 | better |
| synthetic | silicate_4el | median_rel_err | 0.3544 | 0.3015 | -0.05291 | better |
| synthetic | silicate_4el | spurious_mass | 0 | 0 | 0 | same |
| synthetic | silicate_4el | abs_T_rel_err | 0.1056 | 0.06572 | -0.03992 | better |
| synthetic | silicate_4el | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | silicate_4el | dominant_correct | 1 | 1 | 0 | same |
| synthetic | feldspar_k_na | median_rel_err | 0.05007 | 0.1587 | 0.1086 | worse |
| synthetic | feldspar_k_na | spurious_mass | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | abs_T_rel_err | 0.04329 | 0.1146 | 0.07135 | worse |
| synthetic | feldspar_k_na | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | dominant_correct | 1 | 1 | 0 | same |
| synthetic | iron_rich_low_t | median_rel_err | 0.01668 | 0.09999 | 0.08331 | worse |
| synthetic | iron_rich_low_t | spurious_mass | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | abs_T_rel_err | 0.01149 | 0.01416 | 0.002673 | worse |
| synthetic | iron_rich_low_t | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | dominant_correct | 1 | 1 | 0 | same |
| synthetic | binary_argon_ambient | median_rel_err | 0.1921 | 0.1595 | -0.0326 | better |
| synthetic | binary_argon_ambient | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | abs_T_rel_err | 0.1112 | 0.07392 | -0.03725 | better |
| synthetic | binary_argon_ambient | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | dominant_correct | 1 | 1 | 0 | same |
| real | REE_01 | r_squared | -0.2884 | -0.278 | 0.01046 | better |
| real | REE_01 | median_stage_disagreement | 0.4255 | 0.5248 | 0.09925 | worse |
| real | REE_01 | n_detected | 10 | 10 | 0 | same |
| real | REE_01 | ne_at_bound | 1 | 1 | 0 | same |
| real | REE_44 | r_squared | 0.6675 | 0.738 | 0.07053 | better |
| real | REE_44 | median_stage_disagreement | 0.4773 | 0.3974 | -0.07983 | better |
| real | REE_44 | n_detected | 13 | 13 | 0 | same |
| real | REE_44 | ne_at_bound | 1 | 1 | 0 | same |
| real | argon_noAr | r_squared | 0.3048 | 0.3406 | 0.03576 | better |
| real | argon_noAr | median_stage_disagreement | 0.3789 | 0.3811 | 0.002224 | worse |
| real | argon_noAr | n_detected | 9 | 9 | 0 | same |
| real | argon_noAr | ne_at_bound | 1 | 1 | 0 | same |
| real | scan9x9 | r_squared | 0.8449 | 0.8298 | -0.0151 | worse |
| real | scan9x9 | median_stage_disagreement | 0.6548 | 0.4637 | -0.1911 | better |
| real | scan9x9 | n_detected | 16 | 16 | 0 | same |
| real | scan9x9 | ne_at_bound | 1 | 1 | 0 | same |
| real | user_spec1 | r_squared | 0.9135 | 0.9158 | 0.002258 | better |
| real | user_spec1 | median_stage_disagreement | 0.5082 | 0.6006 | 0.09244 | worse |
| real | user_spec1 | n_detected | 10 | 10 | 0 | same |
| real | user_spec1 | ne_at_bound | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | fe_fraction | 0.9021 | 0.6561 | -0.2459 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | non_fe_mass | 0.09795 | 0.3439 | 0.2459 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | abs_T_dev_10kK | 1.126 | 1.092 | -0.03411 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | fe_fraction | 0.01851 | 0.016 | -0.00251 | worse |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | non_fe_mass | 0.9815 | 0.984 | 0.00251 | worse |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | abs_T_dev_10kK | 0.07936 | 0.05149 | -0.02787 | better |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | fe_fraction | 0.9209 | 0.9341 | 0.01318 | better |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | non_fe_mass | 0.0791 | 0.06593 | -0.01318 | better |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | abs_T_dev_10kK | 1.229 | 1.316 | 0.08715 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | fe_fraction | 0.9165 | 0.8958 | -0.02071 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | non_fe_mass | 0.08349 | 0.1042 | 0.02071 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | n_non_fe | 1 | 1 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | abs_T_dev_10kK | 1.042 | 1.157 | 0.115 | worse |
| fe | run-5291e643082343619a4973a1b67898db | fe_fraction | 0.2428 | 0.08353 | -0.1592 | worse |
| fe | run-5291e643082343619a4973a1b67898db | non_fe_mass | 0.7572 | 0.9165 | 0.1592 | worse |
| fe | run-5291e643082343619a4973a1b67898db | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-5291e643082343619a4973a1b67898db | abs_T_dev_10kK | 1.386 | 1.391 | 0.004609 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | fe_fraction | 0.9363 | 0.9611 | 0.02482 | better |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | non_fe_mass | 0.06374 | 0.03892 | -0.02482 | better |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | n_non_fe | 1 | 1 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | abs_T_dev_10kK | 1.313 | 1.382 | 0.06866 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | fe_fraction | 0.1034 | 0.0809 | -0.02251 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | non_fe_mass | 0.8966 | 0.9191 | 0.02251 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | abs_T_dev_10kK | 1.376 | 1.391 | 0.01464 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | fe_fraction | 0.9922 | 0.9689 | -0.02326 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | non_fe_mass | 0.007801 | 0.03106 | 0.02326 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | abs_T_dev_10kK | 1.437 | 1.391 | -0.04609 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | fe_fraction | 0.955 | 0.9417 | -0.01324 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | non_fe_mass | 0.04504 | 0.05829 | 0.01324 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | abs_T_dev_10kK | 1.151 | 1.222 | 0.07138 | worse |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | fe_fraction | 0.02931 | 0.03066 | 0.001353 | better |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | non_fe_mass | 0.9707 | 0.9693 | -0.001353 | better |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | abs_T_dev_10kK | 1.274 | 1.285 | 0.01055 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | fe_fraction | 0.007641 | 0.007371 | -0.0002695 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | non_fe_mass | 0.9924 | 0.9926 | 0.0002695 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | abs_T_dev_10kK | 1.146 | 1.095 | -0.05147 | better |
| fe | run-39998be9f14c479e8008efecec3bee25 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | fe_fraction | 0.06434 | 0.07048 | 0.006149 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | non_fe_mass | 0.9357 | 0.9295 | -0.006149 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | abs_T_dev_10kK | 1.434 | 1.39 | -0.04425 | better |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | fe_fraction | 0.9922 | 0.9731 | -0.01911 | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | non_fe_mass | 0.007764 | 0.02687 | 0.01911 | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | abs_T_dev_10kK | 1.5 | 1.437 | -0.06348 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | fe_fraction | 0.04942 | 0.9723 | 0.9228 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | non_fe_mass | 0.9506 | 0.02773 | -0.9228 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | abs_T_dev_10kK | 1.391 | 1.433 | 0.0417 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | fe_fraction | 0.9246 | 0.932 | 0.007383 | better |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | non_fe_mass | 0.07539 | 0.06801 | -0.007383 | better |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | abs_T_dev_10kK | 1.24 | 1.213 | -0.02752 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | fe_fraction | 0.93 | 0.9548 | 0.02473 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | non_fe_mass | 0.06998 | 0.04525 | -0.02473 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | abs_T_dev_10kK | 1.233 | 1.333 | 0.1005 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | fe_fraction | 0.253 | 0 | -0.253 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | non_fe_mass | 0.747 | 1 | 0.253 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | fe_fraction | 0.9866 | 0.04546 | -0.9411 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | non_fe_mass | 0.01341 | 0.9545 | 0.9411 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | n_non_fe | 1 | 1 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | fe_fraction | 0.05444 | 0.05221 | -0.002229 | worse |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | non_fe_mass | 0.9456 | 0.9478 | 0.002229 | worse |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | abs_T_dev_10kK | 1.437 | 1.435 | -0.001531 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | fe_fraction | 0.7915 | 1 | 0.2085 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | non_fe_mass | 0.2085 | 0 | -0.2085 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | n_non_fe | 2 | 0 | -2 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | abs_T_dev_10kK | 1.304 | 1.233 | -0.07019 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | fe_fraction | 0.6496 | 0.7028 | 0.05323 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | non_fe_mass | 0.3504 | 0.2972 | -0.05323 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | n_non_fe | 2 | 2 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | abs_T_dev_10kK | 1.171 | 1.171 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | fe_fraction | 0.9811 | 0.9806 | -0.0005504 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | non_fe_mass | 0.01888 | 0.01943 | 0.0005504 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | abs_T_dev_10kK | 1.437 | 1.388 | -0.04877 | better |
| fe | run-90143c6b30884618ade5aea2f14819b8 | fe_fraction | 0.9932 | 0.9854 | -0.007791 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | non_fe_mass | 0.00685 | 0.01464 | 0.007791 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | abs_T_dev_10kK | 1.437 | 1.437 | -7.172e-06 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | fe_fraction | 0.9315 | 0.9349 | 0.003422 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | non_fe_mass | 0.06848 | 0.06506 | -0.003422 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | n_non_fe | 1 | 1 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | abs_T_dev_10kK | 1.214 | 1.213 | -0.001026 | better |

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 3 | 2 | 0 | 0 | 0.1697 | 0.1741 |
| synthetic | spurious_mass | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | abs_T_rel_err | 3 | 2 | 0 | 0 | 0.07663 | 0.06848 |
| synthetic | n_majors_missed | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | dominant_correct | 1 | 0 | 4 | 0 | 0.8 | 1 |
| real | r_squared | 4 | 1 | 0 | 0 | 0.4885 | 0.5092 |
| real | median_stage_disagreement | 2 | 3 | 0 | 0 | 0.4889 | 0.4735 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.6 | 11.6 |
| real | ne_at_bound | 0 | 0 | 5 | 0 | 1 | 1 |
| fe | fe_fraction | 10 | 14 | 2 | 0 | 0.5625 | 0.5454 |
| fe | non_fe_mass | 10 | 14 | 2 | 0 | 0.4375 | 0.4546 |
| fe | n_non_fe | 1 | 3 | 22 | 0 | 1.077 | 1.115 |
| fe | abs_T_dev_10kK | 12 | 9 | 5 | 0 | 1.283 | 1.287 |

## Verdict

- **synthetic**: improved (7 better vs 4 worse metric-comparisons)
- **real**: improved (6 better vs 4 worse metric-comparisons)
- **fe**: regressed (33 better vs 40 worse metric-comparisons)
