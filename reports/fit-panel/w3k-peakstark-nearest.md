# Fit-quality comparison: w3k-base -> w3k-peakstark-nearest

baseline config: `{}`  
variant config: `{"overlap_kernel": "peak+stark", "overlap_mode": "nearest"}`

## Per-item

| panel | item | metric | before | after | delta | verdict |
|---|---|---|---|---|---|---|
| synthetic | binary_ca_mg | median_rel_err | 0.1666 | 0.1512 | -0.01541 | better |
| synthetic | binary_ca_mg | spurious_mass | 0 | 0.1786 | 0.1786 | worse |
| synthetic | binary_ca_mg | abs_T_rel_err | 0.1117 | 0.05494 | -0.05672 | better |
| synthetic | binary_ca_mg | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | dominant_correct | 1 | 1 | 0 | same |
| synthetic | silicate_4el | median_rel_err | 0.3615 | 0.8295 | 0.468 | worse |
| synthetic | silicate_4el | spurious_mass | 0 | 0.00152 | 0.00152 | worse |
| synthetic | silicate_4el | abs_T_rel_err | 0.1048 | 0.03518 | -0.06967 | better |
| synthetic | silicate_4el | n_majors_missed | 0 | 1 | 1 | worse |
| synthetic | silicate_4el | dominant_correct | 1 | 1 | 0 | same |
| synthetic | feldspar_k_na | median_rel_err | 0.08967 | 0.3821 | 0.2924 | worse |
| synthetic | feldspar_k_na | spurious_mass | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | abs_T_rel_err | 0.01215 | 0.06615 | 0.05399 | worse |
| synthetic | feldspar_k_na | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | dominant_correct | 1 | 0 | -1 | worse |
| synthetic | iron_rich_low_t | median_rel_err | 0.009964 | 0.07097 | 0.06101 | worse |
| synthetic | iron_rich_low_t | spurious_mass | 0 | 0.0008445 | 0.0008445 | worse |
| synthetic | iron_rich_low_t | abs_T_rel_err | 0.001947 | 0.03381 | 0.03186 | worse |
| synthetic | iron_rich_low_t | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | dominant_correct | 1 | 1 | 0 | same |
| synthetic | binary_argon_ambient | median_rel_err | 0.152 | 0.191 | 0.03895 | worse |
| synthetic | binary_argon_ambient | spurious_mass | 0 | 0.2217 | 0.2217 | worse |
| synthetic | binary_argon_ambient | abs_T_rel_err | 0.1125 | 0.05533 | -0.05718 | better |
| synthetic | binary_argon_ambient | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | dominant_correct | 1 | 1 | 0 | same |
| real | REE_01 | r_squared | -0.2883 | -0.3035 | -0.01524 | worse |
| real | REE_01 | median_stage_disagreement | 0.4263 | 0.6832 | 0.2569 | worse |
| real | REE_01 | n_detected | 10 | 10 | 0 | same |
| real | REE_01 | ne_at_bound | 1 | 1 | 0 | same |
| real | REE_44 | r_squared | 0.6625 | 0.7467 | 0.0842 | better |
| real | REE_44 | median_stage_disagreement | 0.4737 | 0.4948 | 0.02105 | worse |
| real | REE_44 | n_detected | 13 | 17 | 4 | changed |
| real | REE_44 | ne_at_bound | 1 | 1 | 0 | same |
| real | argon_noAr | r_squared | 0.2627 | 0.3101 | 0.04736 | better |
| real | argon_noAr | median_stage_disagreement | 0.4241 | 0.4603 | 0.03615 | worse |
| real | argon_noAr | n_detected | 9 | 9 | 0 | same |
| real | argon_noAr | ne_at_bound | 1 | 1 | 0 | same |
| real | scan9x9 | r_squared | 0.8305 | 0.7626 | -0.06799 | worse |
| real | scan9x9 | median_stage_disagreement | 0.6732 | 0.6271 | -0.04605 | better |
| real | scan9x9 | n_detected | 17 | 16 | -1 | changed |
| real | scan9x9 | ne_at_bound | 1 | 1 | 0 | same |
| real | user_spec1 | r_squared | 0.9235 | 0.8436 | -0.07996 | worse |
| real | user_spec1 | median_stage_disagreement | 0.501 | 0.5503 | 0.0493 | worse |
| real | user_spec1 | n_detected | 10 | 10 | 0 | same |
| real | user_spec1 | ne_at_bound | 1 | 0 | -1 | better |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | fe_fraction | 0.9111 | 0.9473 | 0.03624 | better |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | non_fe_mass | 0.08891 | 0.05267 | -0.03624 | better |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | abs_T_dev_10kK | 1.145 | 1.092 | -0.05321 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | fe_fraction | 0.01915 | 0.03542 | 0.01627 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | non_fe_mass | 0.9808 | 0.9646 | -0.01627 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | abs_T_dev_10kK | 0.07936 | 0.0004567 | -0.0789 | better |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | fe_fraction | 0.9643 | 0.9289 | -0.03548 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | non_fe_mass | 0.03567 | 0.07115 | 0.03548 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | abs_T_dev_10kK | 1.231 | 1.391 | 0.1602 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | fe_fraction | 0.9187 | 0.9407 | 0.02206 | better |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | non_fe_mass | 0.08133 | 0.05927 | -0.02206 | better |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | n_non_fe | 1 | 1 | 0 | same |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | abs_T_dev_10kK | 1.047 | 1.264 | 0.2171 | worse |
| fe | run-5291e643082343619a4973a1b67898db | fe_fraction | 0.07293 | 0.3484 | 0.2755 | better |
| fe | run-5291e643082343619a4973a1b67898db | non_fe_mass | 0.9271 | 0.6516 | -0.2755 | better |
| fe | run-5291e643082343619a4973a1b67898db | n_non_fe | 2 | 2 | 0 | same |
| fe | run-5291e643082343619a4973a1b67898db | abs_T_dev_10kK | 1.294 | 1.391 | 0.09665 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | fe_fraction | 0.9711 | 0.9909 | 0.01986 | better |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | non_fe_mass | 0.02891 | 0.009056 | -0.01986 | better |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | n_non_fe | 1 | 1 | 0 | same |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | abs_T_dev_10kK | 1.32 | 1.391 | 0.07045 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | fe_fraction | 0.105 | 0.03163 | -0.07335 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | non_fe_mass | 0.895 | 0.9684 | 0.07335 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | abs_T_dev_10kK | 1.391 | 1.391 | -3.91e-05 | better |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | fe_fraction | 0.9721 | 0.9925 | 0.02047 | better |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | non_fe_mass | 0.02794 | 0.007476 | -0.02047 | better |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | abs_T_dev_10kK | 1.437 | 1.37 | -0.06686 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | fe_fraction | 0.8652 | 0.9954 | 0.1301 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | non_fe_mass | 0.1348 | 0.004613 | -0.1301 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | n_non_fe | 1 | 0 | -1 | better |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | abs_T_dev_10kK | 1.092 | 1.417 | 0.325 | worse |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | abs_T_dev_10kK | 1.5 | 0.5333 | -0.9667 | better |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | fe_fraction | 0.05806 | 0.03645 | -0.02162 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | non_fe_mass | 0.9419 | 0.9636 | 0.02162 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | abs_T_dev_10kK | 1.248 | 1.204 | -0.04386 | better |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | fe_fraction | 0.007641 | 0.00967 | 0.002029 | better |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | non_fe_mass | 0.9924 | 0.9903 | -0.002029 | better |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | abs_T_dev_10kK | 1.146 | 1.386 | 0.2395 | worse |
| fe | run-39998be9f14c479e8008efecec3bee25 | fe_fraction | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | n_non_fe | 0 | 0 | 0 | same |
| fe | run-39998be9f14c479e8008efecec3bee25 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | fe_fraction | 0.06471 | 0.0907 | 0.026 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | non_fe_mass | 0.9353 | 0.9093 | -0.026 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | abs_T_dev_10kK | 1.437 | 1.436 | -0.001165 | better |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | fe_fraction | 0.9922 | 0.9939 | 0.001693 | better |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | non_fe_mass | 0.007764 | 0.006071 | -0.001693 | better |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | abs_T_dev_10kK | 1.5 | 1.437 | -0.06302 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | fe_fraction | 0.9541 | 0.9779 | 0.02378 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | non_fe_mass | 0.04592 | 0.02215 | -0.02378 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | abs_T_dev_10kK | 1.349 | 1.428 | 0.0791 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | fe_fraction | 0.9246 | 0.9832 | 0.05859 | better |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | non_fe_mass | 0.07538 | 0.01679 | -0.05859 | better |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | abs_T_dev_10kK | 1.244 | 1.437 | 0.1924 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | fe_fraction | 0.9242 | 0.9726 | 0.04833 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | non_fe_mass | 0.07577 | 0.02743 | -0.04833 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | abs_T_dev_10kK | 1.233 | 1.376 | 0.1425 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | fe_fraction | 0 | 0 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | non_fe_mass | 1 | 1 | 0 | same |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | n_non_fe | 2 | 3 | 1 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | fe_fraction | 0.05329 | 0.04149 | -0.0118 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | non_fe_mass | 0.9467 | 0.9585 | 0.0118 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | n_non_fe | 1 | 1 | 0 | same |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | fe_fraction | 0.05444 | 0.05318 | -0.001262 | worse |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | non_fe_mass | 0.9456 | 0.9468 | 0.001262 | worse |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | abs_T_dev_10kK | 1.437 | 1.437 | -0.0001471 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | fe_fraction | 0.7915 | 1 | 0.2085 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | non_fe_mass | 0.2085 | 0 | -0.2085 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | n_non_fe | 2 | 0 | -2 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | abs_T_dev_10kK | 1.304 | 1.391 | 0.08719 | worse |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | fe_fraction | 0.6494 | 0.7162 | 0.06676 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | non_fe_mass | 0.3506 | 0.2838 | -0.06676 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | n_non_fe | 2 | 2 | 0 | same |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | abs_T_dev_10kK | 1.168 | 1.18 | 0.01162 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | fe_fraction | 0.9947 | 0.9955 | 0.0007377 | better |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | non_fe_mass | 0.005264 | 0.004527 | -0.0007377 | better |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | n_non_fe | 1 | 0 | -1 | better |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | abs_T_dev_10kK | 1.437 | 1.391 | -0.0459 | better |
| fe | run-90143c6b30884618ade5aea2f14819b8 | fe_fraction | 0.9927 | 0.9902 | -0.002504 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | non_fe_mass | 0.007342 | 0.009845 | 0.002504 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | abs_T_dev_10kK | 1.437 | 1.437 | 5.737e-05 | worse |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | fe_fraction | 0.9257 | 0.9392 | 0.01352 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | non_fe_mass | 0.07427 | 0.06076 | -0.01352 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | n_non_fe | 1 | 1 | 0 | same |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | abs_T_dev_10kK | 1.187 | 1.163 | -0.02332 | better |

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 1 | 4 | 0 | 0 | 0.1559 | 0.3249 |
| synthetic | spurious_mass | 0 | 4 | 1 | 0 | 0 | 0.08054 |
| synthetic | abs_T_rel_err | 3 | 2 | 0 | 0 | 0.06862 | 0.04908 |
| synthetic | n_majors_missed | 0 | 1 | 4 | 0 | 0 | 0.2 |
| synthetic | dominant_correct | 0 | 1 | 4 | 0 | 1 | 0.8 |
| real | r_squared | 2 | 3 | 0 | 0 | 0.4782 | 0.4719 |
| real | median_stage_disagreement | 1 | 4 | 0 | 0 | 0.4997 | 0.5631 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.8 | 12.4 |
| real | ne_at_bound | 1 | 0 | 4 | 0 | 1 | 0.8 |
| fe | fe_fraction | 17 | 6 | 3 | 0 | 0.5456 | 0.5774 |
| fe | non_fe_mass | 17 | 6 | 3 | 0 | 0.4544 | 0.4226 |
| fe | n_non_fe | 3 | 4 | 19 | 0 | 1.154 | 1.154 |
| fe | abs_T_dev_10kK | 11 | 12 | 3 | 0 | 1.276 | 1.286 |

## Verdict

- **synthetic**: regressed (4 better vs 12 worse metric-comparisons)
- **real**: regressed (4 better vs 7 worse metric-comparisons)
- **fe**: improved (48 better vs 28 worse metric-comparisons)
