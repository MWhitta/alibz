# Fit-quality comparison: gate-base -> gate-fix

baseline config: `{}`  
variant config: `{}`

## Per-item

| panel | item | metric | before | after | delta | verdict |
|---|---|---|---|---|---|---|
| synthetic | binary_ca_mg | median_rel_err | 0.2352 | 0.2352 | 0 | same |
| synthetic | binary_ca_mg | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | abs_T_rel_err | 0.1116 | 0.1116 | 0 | same |
| synthetic | binary_ca_mg | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_ca_mg | dominant_correct | 0 | 0 | 0 | same |
| synthetic | silicate_4el | median_rel_err | 0.3544 | 0.3544 | 0 | same |
| synthetic | silicate_4el | spurious_mass | 0 | 0 | 0 | same |
| synthetic | silicate_4el | abs_T_rel_err | 0.1056 | 0.1056 | 0 | same |
| synthetic | silicate_4el | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | silicate_4el | dominant_correct | 1 | 1 | 0 | same |
| synthetic | feldspar_k_na | median_rel_err | 0.05007 | 0.05007 | 0 | same |
| synthetic | feldspar_k_na | spurious_mass | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | abs_T_rel_err | 0.04329 | 0.04329 | 0 | same |
| synthetic | feldspar_k_na | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | feldspar_k_na | dominant_correct | 1 | 1 | 0 | same |
| synthetic | iron_rich_low_t | median_rel_err | 0.01668 | 0.01668 | 0 | same |
| synthetic | iron_rich_low_t | spurious_mass | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | abs_T_rel_err | 0.01149 | 0.01149 | 0 | same |
| synthetic | iron_rich_low_t | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | iron_rich_low_t | dominant_correct | 1 | 1 | 0 | same |
| synthetic | binary_argon_ambient | median_rel_err | 0.1921 | 0.1921 | 0 | same |
| synthetic | binary_argon_ambient | spurious_mass | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | abs_T_rel_err | 0.1112 | 0.1112 | 0 | same |
| synthetic | binary_argon_ambient | n_majors_missed | 0 | 0 | 0 | same |
| synthetic | binary_argon_ambient | dominant_correct | 1 | 1 | 0 | same |
| real | REE_01 | r_squared | -0.2884 | -0.2884 | 0 | same |
| real | REE_01 | median_stage_disagreement | 0.4255 | 0.4255 | 0 | same |
| real | REE_01 | n_detected | 10 | 10 | 0 | same |
| real | REE_01 | ne_at_bound | 1 | 1 | 0 | same |
| real | REE_44 | r_squared | 0.6675 | 0.6675 | 0 | same |
| real | REE_44 | median_stage_disagreement | 0.4773 | 0.4773 | 0 | same |
| real | REE_44 | n_detected | 13 | 13 | 0 | same |
| real | REE_44 | ne_at_bound | 1 | 1 | 0 | same |
| real | argon_noAr | r_squared | 0.3048 | 0.3048 | 0 | same |
| real | argon_noAr | median_stage_disagreement | 0.3789 | 0.3789 | 0 | same |
| real | argon_noAr | n_detected | 9 | 9 | 0 | same |
| real | argon_noAr | ne_at_bound | 1 | 1 | 0 | same |
| real | scan9x9 | r_squared | 0.8449 | 0.8449 | 0 | same |
| real | scan9x9 | median_stage_disagreement | 0.6548 | 0.6548 | 0 | same |
| real | scan9x9 | n_detected | 16 | 16 | 0 | same |
| real | scan9x9 | ne_at_bound | 1 | 1 | 0 | same |
| real | user_spec1 | r_squared | 0.9135 | 0.9135 | 0 | same |
| real | user_spec1 | median_stage_disagreement | 0.5082 | 0.5082 | 0 | same |
| real | user_spec1 | n_detected | 10 | 10 | 0 | same |
| real | user_spec1 | ne_at_bound | 1 | 1 | 0 | same |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | fe_fraction | 0.9021 | 0.06041 | -0.8416 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | non_fe_mass | 0.09795 | 0.9396 | 0.8416 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-e825d9f567204c5cbc6fc33c9c16bbf7 | abs_T_dev_10kK | 1.126 | 1.5 | 0.3742 | worse |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | fe_fraction | 0.01851 | 0.7624 | 0.7438 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | non_fe_mass | 0.9815 | 0.2376 | -0.7438 | better |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | n_non_fe | 2 | 2 | 0 | same |
| fe | run-a50ac639cc7747b4af81a677c8e99312 | abs_T_dev_10kK | 0.07936 | 0.06288 | -0.01648 | better |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | fe_fraction | 0.9209 | 0.00848 | -0.9124 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | non_fe_mass | 0.0791 | 0.9915 | 0.9124 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-eae9d54f22834928ae2f6af8d4a23779 | abs_T_dev_10kK | 1.229 | 1.421 | 0.1918 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | fe_fraction | 0.9165 | 0.3959 | -0.5206 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | non_fe_mass | 0.08349 | 0.6041 | 0.5206 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-126d0c96e1584131a0f8d7e5fb065f1a | abs_T_dev_10kK | 1.042 | 1.401 | 0.3582 | worse |
| fe | run-5291e643082343619a4973a1b67898db | fe_fraction | 0.2428 | 0.03466 | -0.2081 | worse |
| fe | run-5291e643082343619a4973a1b67898db | non_fe_mass | 0.7572 | 0.9653 | 0.2081 | worse |
| fe | run-5291e643082343619a4973a1b67898db | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-5291e643082343619a4973a1b67898db | abs_T_dev_10kK | 1.386 | 1.187 | -0.1994 | better |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | fe_fraction | 0.9363 | 0.5481 | -0.3881 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | non_fe_mass | 0.06374 | 0.4519 | 0.3881 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-11d07e557f4743c290137c2dcdf6fdfc | abs_T_dev_10kK | 1.313 | 1.204 | -0.1092 | better |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | fe_fraction | 0.1034 | 0.6293 | 0.5259 | better |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | non_fe_mass | 0.8966 | 0.3707 | -0.5259 | better |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-6e38b110215a43db86ba3d81273f67f3 | abs_T_dev_10kK | 1.376 | 1.5 | 0.1238 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | fe_fraction | 0.9922 | 0 | -0.9922 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | non_fe_mass | 0.007801 | 1 | 0.9922 | worse |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-c52b7f2159244dc2b7bf16a33dc5c346 | abs_T_dev_10kK | 1.437 | 1.5 | 0.06325 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | fe_fraction | 0.955 | 0.4999 | -0.4551 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | non_fe_mass | 0.04504 | 0.5001 | 0.4551 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-b95ffbb4f6ac4fc6944e0cb9d9e26b61 | abs_T_dev_10kK | 1.151 | 1.088 | -0.06285 | better |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | fe_fraction | 0 | 0.5362 | 0.5362 | better |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | non_fe_mass | 1 | 0.4638 | -0.5362 | better |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | n_non_fe | 0 | 2 | 2 | worse |
| fe | run-674f4feb6d2149c4bac365e395a4d327 | abs_T_dev_10kK | 1.5 | 1.245 | -0.2551 | better |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | fe_fraction | 0.02931 | 0.57 | 0.5407 | better |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | non_fe_mass | 0.9707 | 0.43 | -0.5407 | better |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-56a8de4a62864db0865c87c1d6b408b8 | abs_T_dev_10kK | 1.274 | 1.258 | -0.01646 | better |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | fe_fraction | 0.007641 | 0 | -0.007641 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | non_fe_mass | 0.9924 | 1 | 0.007641 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-da01e553ae6a4c91995aa2d403904fa2 | abs_T_dev_10kK | 1.146 | 1.5 | 0.3539 | worse |
| fe | run-39998be9f14c479e8008efecec3bee25 | fe_fraction | 0 | 0.6019 | 0.6019 | better |
| fe | run-39998be9f14c479e8008efecec3bee25 | non_fe_mass | 1 | 0.3981 | -0.6019 | better |
| fe | run-39998be9f14c479e8008efecec3bee25 | n_non_fe | 0 | 2 | 2 | worse |
| fe | run-39998be9f14c479e8008efecec3bee25 | abs_T_dev_10kK | 1.5 | 1.177 | -0.3226 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | fe_fraction | 0.06434 | 0.7436 | 0.6793 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | non_fe_mass | 0.9357 | 0.2564 | -0.6793 | better |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-8264fabf6e4d4e61b05df90ef45cecb3 | abs_T_dev_10kK | 1.434 | 0.006078 | -1.428 | better |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | fe_fraction | 0.9922 | 0 | -0.9922 | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | non_fe_mass | 0.007764 | 1 | 0.9922 | worse |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-b17cc0430a5c41c097fcfde0ca73bbf4 | abs_T_dev_10kK | 1.5 | 1.5 | 0 | same |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | fe_fraction | 0.04942 | 0.4137 | 0.3643 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | non_fe_mass | 0.9506 | 0.5863 | -0.3643 | better |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-d99401be15bb4ca19d5b59e0f61a57e5 | abs_T_dev_10kK | 1.391 | 1.261 | -0.1295 | better |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | fe_fraction | 0.9246 | 0 | -0.9246 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | non_fe_mass | 0.07539 | 1 | 0.9246 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | n_non_fe | 1 | 4 | 3 | worse |
| fe | run-bb777df5e62c4fc18c3e2c33cbb6c766 | abs_T_dev_10kK | 1.24 | 0.1712 | -1.069 | better |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | fe_fraction | 0.93 | 0.1176 | -0.8124 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | non_fe_mass | 0.06998 | 0.8824 | 0.8124 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-07c25af6dd9b4fa6bc801e18a40661a4 | abs_T_dev_10kK | 1.233 | 1.067 | -0.1655 | better |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | fe_fraction | 0.253 | 0.0128 | -0.2402 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | non_fe_mass | 0.747 | 0.9872 | 0.2402 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-6cd149cca8d648469e3486cf60f5ee5c | abs_T_dev_10kK | 1.5 | 0.9692 | -0.5308 | better |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | fe_fraction | 0.9866 | 0.008618 | -0.978 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | non_fe_mass | 0.01341 | 0.9914 | 0.978 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-2e03f73db1f441939ff7bf2480d75e0d | abs_T_dev_10kK | 1.5 | 1.334 | -0.166 | better |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | fe_fraction | 0.05444 | 0.2435 | 0.1891 | better |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | non_fe_mass | 0.9456 | 0.7565 | -0.1891 | better |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | n_non_fe | 2 | 1 | -1 | better |
| fe | run-6ce8e9b3945f4429997104a9b1fa22f9 | abs_T_dev_10kK | 1.437 | 1.393 | -0.04392 | better |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | fe_fraction | 0.7915 | 0.1191 | -0.6724 | worse |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | non_fe_mass | 0.2085 | 0.8809 | 0.6724 | worse |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | n_non_fe | 2 | 3 | 1 | worse |
| fe | run-468a150cbd4a458a8fa0fda3b09432dc | abs_T_dev_10kK | 1.304 | 1.237 | -0.06672 | better |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | fe_fraction | 0.6496 | 0.08068 | -0.5689 | worse |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | non_fe_mass | 0.3504 | 0.9193 | 0.5689 | worse |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | n_non_fe | 2 | 3 | 1 | worse |
| fe | run-fd1e6d48a1034cd09e6780a94f43111b | abs_T_dev_10kK | 1.171 | 1.437 | 0.2663 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | fe_fraction | 0.9811 | 0.0109 | -0.9702 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | non_fe_mass | 0.01888 | 0.9891 | 0.9702 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | n_non_fe | 1 | 3 | 2 | worse |
| fe | run-01105eaef0044dca87fb08bfdea5b107 | abs_T_dev_10kK | 1.437 | 1.179 | -0.2576 | better |
| fe | run-90143c6b30884618ade5aea2f14819b8 | fe_fraction | 0.9932 | 0.9858 | -0.007357 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | non_fe_mass | 0.00685 | 0.01421 | 0.007357 | worse |
| fe | run-90143c6b30884618ade5aea2f14819b8 | n_non_fe | 1 | 1 | 0 | same |
| fe | run-90143c6b30884618ade5aea2f14819b8 | abs_T_dev_10kK | 1.437 | 1.3 | -0.1366 | better |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | fe_fraction | 0.9315 | 0.06644 | -0.8651 | worse |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | non_fe_mass | 0.06848 | 0.9336 | 0.8651 | worse |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | n_non_fe | 1 | 2 | 1 | worse |
| fe | run-74c9332760af4a1ab6f4a343c5e47c4f | abs_T_dev_10kK | 1.214 | 1.209 | -0.004846 | better |

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 0 | 0 | 5 | 0 | 0.1697 | 0.1697 |
| synthetic | spurious_mass | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | abs_T_rel_err | 0 | 0 | 5 | 0 | 0.07663 | 0.07663 |
| synthetic | n_majors_missed | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | dominant_correct | 0 | 0 | 5 | 0 | 0.8 | 0.8 |
| real | r_squared | 0 | 0 | 5 | 0 | 0.4885 | 0.4885 |
| real | median_stage_disagreement | 0 | 0 | 5 | 0 | 0.4889 | 0.4889 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.6 | 11.6 |
| real | ne_at_bound | 0 | 0 | 5 | 0 | 1 | 1 |
| fe | fe_fraction | 8 | 18 | 0 | 0 | 0.5625 | 0.2865 |
| fe | non_fe_mass | 8 | 18 | 0 | 0 | 0.4375 | 0.7135 |
| fe | n_non_fe | 1 | 21 | 4 | 0 | 1.077 | 2.346 |
| fe | abs_T_dev_10kK | 18 | 7 | 1 | 0 | 1.283 | 1.158 |

## Verdict

- **synthetic**: no net change (0 better vs 0 worse metric-comparisons)
- **real**: no net change (0 better vs 0 worse metric-comparisons)
- **fe**: regressed (35 better vs 64 worse metric-comparisons)

