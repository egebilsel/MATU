====================================================================================================
HARMBENCH MATU GÜVENLIK BELİRSİZLİK ANALİZİ
====================================================================================================

--- Qwen2.5-Coder-3B-Instruct (Base) (harmbench_3b_base) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.5355
  Genel Median Unc.       : 4.5902
  SAFE  Sorular Unc.      : 4.5604
  UNSAFE Sorular Unc.     : 4.2056
  GAP (Unsafe - Safe)     : -0.3548  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4220
  AUROC (majority_incorr) : 0.2704
  AUARC                   : 0.8236

--- AZR-Coder-3b (Evolved) (harmbench_3b_evolved) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.1552
  Genel Median Unc.       : 4.1292
  SAFE  Sorular Unc.      : 4.4121
  UNSAFE Sorular Unc.     : 3.7699
  GAP (Unsafe - Safe)     : -0.6422  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.1733
  AUROC (majority_incorr) : 0.1950
  AUARC                   : 0.3850

--- Qwen2.5-Coder-7B-Instruct (Base) (harmbench_7b_base) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 96.9%
  Safe Behavior (majority): 99/100
  Unsafe Behavior         : 1/100

  Genel Ortalama Unc.     : 3.7636
  Genel Median Unc.       : 4.0708
  SAFE  Sorular Unc.      : 3.7567
  UNSAFE Sorular Unc.     : 4.4453
  GAP (Unsafe - Safe)     : 0.6886  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.7807
  AUROC (majority_incorr) : 0.7980
  AUARC                   : 0.9811

--- AZR-Coder-7b (Evolved) (harmbench_7b_evolved) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 65.6%
  Safe Behavior (majority): 73/100
  Unsafe Behavior         : 27/100

  Genel Ortalama Unc.     : 4.1724
  Genel Median Unc.       : 4.1849
  SAFE  Sorular Unc.      : 4.3487
  UNSAFE Sorular Unc.     : 3.6958
  GAP (Unsafe - Safe)     : -0.6529  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.3307
  AUROC (majority_incorr) : 0.2253
  AUARC                   : 0.4784

--- Qwen2.5-Coder-14B-Instruct (Base) (harmbench_14b_base) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 99.6%
  Safe Behavior (majority): 100/100
  Unsafe Behavior         : 0/100

  Genel Ortalama Unc.     : 3.8098
  Genel Median Unc.       : 4.0099
  SAFE  Sorular Unc.      : 3.8098
  UNSAFE Sorular Unc.     : nan
  GAP (Unsafe - Safe)     : nan  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.7109
  AUROC (majority_incorr) : nan
  AUARC                   : 0.9884

--- AZR-Coder-14b (Evolved) (harmbench_14b_evolved) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 60.2%
  Safe Behavior (majority): 64/100
  Unsafe Behavior         : 36/100

  Genel Ortalama Unc.     : 4.0564
  Genel Median Unc.       : 3.9700
  SAFE  Sorular Unc.      : 4.2198
  UNSAFE Sorular Unc.     : 3.7658
  GAP (Unsafe - Safe)     : -0.4540  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.2560
  AUROC (majority_incorr) : 0.2977
  AUARC                   : 0.4438

--- Qwen2.5-7B-Instruct (Base, Non-Coder) (harmbench_7b_base_noncoder) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 89.9%
  Safe Behavior (majority): 94/100
  Unsafe Behavior         : 6/100

  Genel Ortalama Unc.     : 4.1330
  Genel Median Unc.       : 4.0998
  SAFE  Sorular Unc.      : 4.1722
  UNSAFE Sorular Unc.     : 3.5181
  GAP (Unsafe - Safe)     : -0.6542  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.3263
  AUROC (majority_incorr) : 0.1436
  AUARC                   : 0.7974

--- AZR-Base-7b (Evolved, Non-Coder) (harmbench_7b_evolved_noncoder) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 71.6%
  Safe Behavior (majority): 77/100
  Unsafe Behavior         : 23/100

  Genel Ortalama Unc.     : 4.1972
  Genel Median Unc.       : 4.1964
  SAFE  Sorular Unc.      : 4.3138
  UNSAFE Sorular Unc.     : 3.8067
  GAP (Unsafe - Safe)     : -0.5072  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.3366
  AUROC (majority_incorr) : 0.2422
  AUARC                   : 0.5813

--- 3B Base (Cleaned) (harmbench_3b_base_cleaned) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.6973
  Genel Median Unc.       : 4.7206
  SAFE  Sorular Unc.      : 4.7103
  UNSAFE Sorular Unc.     : 4.5250
  GAP (Unsafe - Safe)     : -0.1853  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4667
  AUROC (majority_incorr) : 0.3272
  AUARC                   : 0.8502

--- 3B Evolved (Cleaned) (harmbench_3b_evolved_cleaned) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.5407
  Genel Median Unc.       : 4.5371
  SAFE  Sorular Unc.      : 4.6797
  UNSAFE Sorular Unc.     : 4.3320
  GAP (Unsafe - Safe)     : -0.3477  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.2095
  AUROC (majority_incorr) : 0.3158
  AUARC                   : 0.4666

--- 3B Base (Embedding (4B)) (harmbench_3b_base_embedding) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.3078
  Genel Median Unc.       : 4.3385
  SAFE  Sorular Unc.      : 4.3272
  UNSAFE Sorular Unc.     : 4.0496
  GAP (Unsafe - Safe)     : -0.2776  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4253
  AUROC (majority_incorr) : 0.3057
  AUARC                   : 0.8545

--- 3B Evolved (Embedding (4B)) (harmbench_3b_evolved_embedding) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.2388
  Genel Median Unc.       : 4.2080
  SAFE  Sorular Unc.      : 4.4983
  UNSAFE Sorular Unc.     : 3.8496
  GAP (Unsafe - Safe)     : -0.6487  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.1034
  AUROC (majority_incorr) : 0.1546
  AUARC                   : 0.3639

--- 3B Base (Rank 30) (harmbench_3b_base_rank30) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.4682
  Genel Median Unc.       : 4.4877
  SAFE  Sorular Unc.      : 4.4933
  UNSAFE Sorular Unc.     : 4.1346
  GAP (Unsafe - Safe)     : -0.3587  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4286
  AUROC (majority_incorr) : 0.2627
  AUARC                   : 0.8244

--- 3B Evolved (Rank 30) (harmbench_3b_evolved_rank30) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.1012
  Genel Median Unc.       : 4.1050
  SAFE  Sorular Unc.      : 4.3649
  UNSAFE Sorular Unc.     : 3.7056
  GAP (Unsafe - Safe)     : -0.6592  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.1821
  AUROC (majority_incorr) : 0.1842
  AUARC                   : 0.3808

--- 3B Base (Assistant Only) (harmbench_3b_base_assistant_only) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.8422
  Genel Median Unc.       : 4.9545
  SAFE  Sorular Unc.      : 4.8486
  UNSAFE Sorular Unc.     : 4.7566
  GAP (Unsafe - Safe)     : -0.0920  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.5053
  AUROC (majority_incorr) : 0.4639
  AUARC                   : 0.8557

--- 3B Evolved (Assistant Only) (harmbench_3b_evolved_assistant_only) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.3929
  Genel Median Unc.       : 4.4172
  SAFE  Sorular Unc.      : 4.5859
  UNSAFE Sorular Unc.     : 4.1034
  GAP (Unsafe - Safe)     : -0.4826  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.3015
  AUROC (majority_incorr) : 0.2867
  AUARC                   : 0.4386

--- 3B Base (User Only) (harmbench_3b_base_user_only) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 12.8812
  Genel Median Unc.       : 12.0000
  SAFE  Sorular Unc.      : 12.9367
  UNSAFE Sorular Unc.     : 12.1429
  GAP (Unsafe - Safe)     : -0.7939  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.5246
  AUROC (majority_incorr) : 0.6267
  AUARC                   : 0.8997

--- 3B Evolved (User Only) (harmbench_3b_evolved_user_only) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 12.8812
  Genel Median Unc.       : 12.0000
  SAFE  Sorular Unc.      : 11.3667
  UNSAFE Sorular Unc.     : 15.1529
  GAP (Unsafe - Safe)     : 3.7862  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4315
  AUROC (majority_incorr) : 0.6362
  AUARC                   : 0.5855

--- 3B Base (Time (Exp)) (harmbench_3b_base_time_exp) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.5401
  Genel Median Unc.       : 4.5799
  SAFE  Sorular Unc.      : 4.5652
  UNSAFE Sorular Unc.     : 4.2054
  GAP (Unsafe - Safe)     : -0.3599  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4216
  AUROC (majority_incorr) : 0.2704
  AUARC                   : 0.8231

--- 3B Evolved (Time (Exp)) (harmbench_3b_evolved_time_exp) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.1460
  Genel Median Unc.       : 4.1354
  SAFE  Sorular Unc.      : 4.4062
  UNSAFE Sorular Unc.     : 3.7556
  GAP (Unsafe - Safe)     : -0.6506  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.1857
  AUROC (majority_incorr) : 0.1879
  AUARC                   : 0.3828

--- 3B Base (Time (Linear)) (harmbench_3b_base_time_linear) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.5401
  Genel Median Unc.       : 4.5799
  SAFE  Sorular Unc.      : 4.5652
  UNSAFE Sorular Unc.     : 4.2054
  GAP (Unsafe - Safe)     : -0.3599  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4216
  AUROC (majority_incorr) : 0.2704
  AUARC                   : 0.8231

--- 3B Evolved (Time (Linear)) (harmbench_3b_evolved_time_linear) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.1460
  Genel Median Unc.       : 4.1354
  SAFE  Sorular Unc.      : 4.4062
  UNSAFE Sorular Unc.     : 3.7556
  GAP (Unsafe - Safe)     : -0.6506  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.1857
  AUROC (majority_incorr) : 0.1879
  AUARC                   : 0.3828

--- 3B Base (Time (Cutoff)) (harmbench_3b_base_time_cutoff) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.5355
  Genel Median Unc.       : 4.5902
  SAFE  Sorular Unc.      : 4.5604
  UNSAFE Sorular Unc.     : 4.2056
  GAP (Unsafe - Safe)     : -0.3548  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4220
  AUROC (majority_incorr) : 0.2704
  AUARC                   : 0.8236

--- 3B Evolved (Time (Cutoff)) (harmbench_3b_evolved_time_cutoff) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.1552
  Genel Median Unc.       : 4.1292
  SAFE  Sorular Unc.      : 4.4121
  UNSAFE Sorular Unc.     : 3.7699
  GAP (Unsafe - Safe)     : -0.6422  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.1733
  AUROC (majority_incorr) : 0.1950
  AUARC                   : 0.3850

--- 3B Base (Time (Sigmoid)) (harmbench_3b_base_time_sigmoid) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.5401
  Genel Median Unc.       : 4.5799
  SAFE  Sorular Unc.      : 4.5652
  UNSAFE Sorular Unc.     : 4.2054
  GAP (Unsafe - Safe)     : -0.3599  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4216
  AUROC (majority_incorr) : 0.2704
  AUARC                   : 0.8231

--- 3B Evolved (Time (Sigmoid)) (harmbench_3b_evolved_time_sigmoid) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.1460
  Genel Median Unc.       : 4.1354
  SAFE  Sorular Unc.      : 4.4062
  UNSAFE Sorular Unc.     : 3.7556
  GAP (Unsafe - Safe)     : -0.6506  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.1857
  AUROC (majority_incorr) : 0.1879
  AUARC                   : 0.3828

--- 3B Base (Time (Log)) (harmbench_3b_base_time_log) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 88.0%
  Safe Behavior (majority): 93/100
  Unsafe Behavior         : 7/100

  Genel Ortalama Unc.     : 4.5401
  Genel Median Unc.       : 4.5799
  SAFE  Sorular Unc.      : 4.5652
  UNSAFE Sorular Unc.     : 4.2054
  GAP (Unsafe - Safe)     : -0.3599  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4216
  AUROC (majority_incorr) : 0.2704
  AUARC                   : 0.8231

--- 3B Evolved (Time (Log)) (harmbench_3b_evolved_time_log) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 56.4%
  Safe Behavior (majority): 60/100
  Unsafe Behavior         : 40/100

  Genel Ortalama Unc.     : 4.1460
  Genel Median Unc.       : 4.1354
  SAFE  Sorular Unc.      : 4.4062
  UNSAFE Sorular Unc.     : 3.7556
  GAP (Unsafe - Safe)     : -0.6506  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.1857
  AUROC (majority_incorr) : 0.1879
  AUARC                   : 0.3828

====================================================================================================
ÖZET TABLO
====================================================================================================

Deney                                         |  Safe% | #Safe | #Unsafe | Safe Unc | Unsafe Unc |   AUROC |   AUARC |     Gap
------------------------------------------------------------------------------------------------------------------------------
Qwen2.5-Coder-3B-Instruct (Base)              |  88.0% |    93 |       7 |   4.5604 |     4.2056 |  0.4220 |  0.8236 | -0.3548
AZR-Coder-3b (Evolved)                        |  56.4% |    60 |      40 |   4.4121 |     3.7699 |  0.1733 |  0.3850 | -0.6422
Qwen2.5-Coder-7B-Instruct (Base)              |  96.9% |    99 |       1 |   3.7567 |     4.4453 |  0.7807 |  0.9811 |  0.6886
AZR-Coder-7b (Evolved)                        |  65.6% |    73 |      27 |   4.3487 |     3.6958 |  0.3307 |  0.4784 | -0.6529
Qwen2.5-Coder-14B-Instruct (Base)             |  99.6% |   100 |       0 |   3.8098 |        N/A |  0.7109 |  0.9884 |     N/A
AZR-Coder-14b (Evolved)                       |  60.2% |    64 |      36 |   4.2198 |     3.7658 |  0.2560 |  0.4438 | -0.4540
Qwen2.5-7B-Instruct (Base, Non-Coder)         |  89.9% |    94 |       6 |   4.1722 |     3.5181 |  0.3263 |  0.7974 | -0.6542
AZR-Base-7b (Evolved, Non-Coder)              |  71.6% |    77 |      23 |   4.3138 |     3.8067 |  0.3366 |  0.5813 | -0.5072
3B Base (Cleaned)                             |  88.0% |    93 |       7 |   4.7103 |     4.5250 |  0.4667 |  0.8502 | -0.1853
3B Evolved (Cleaned)                          |  56.4% |    60 |      40 |   4.6797 |     4.3320 |  0.2095 |  0.4666 | -0.3477
3B Base (Embedding (4B))                      |  88.0% |    93 |       7 |   4.3272 |     4.0496 |  0.4253 |  0.8545 | -0.2776
3B Evolved (Embedding (4B))                   |  56.4% |    60 |      40 |   4.4983 |     3.8496 |  0.1034 |  0.3639 | -0.6487
3B Base (Rank 30)                             |  88.0% |    93 |       7 |   4.4933 |     4.1346 |  0.4286 |  0.8244 | -0.3587
3B Evolved (Rank 30)                          |  56.4% |    60 |      40 |   4.3649 |     3.7056 |  0.1821 |  0.3808 | -0.6592
3B Base (Assistant Only)                      |  88.0% |    93 |       7 |   4.8486 |     4.7566 |  0.5053 |  0.8557 | -0.0920
3B Evolved (Assistant Only)                   |  56.4% |    60 |      40 |   4.5859 |     4.1034 |  0.3015 |  0.4386 | -0.4826
3B Base (User Only)                           |  88.0% |    93 |       7 |  12.9367 |    12.1429 |  0.5246 |  0.8997 | -0.7939
3B Evolved (User Only)                        |  56.4% |    60 |      40 |  11.3667 |    15.1529 |  0.4315 |  0.5855 |  3.7862
3B Base (Time (Exp))                          |  88.0% |    93 |       7 |   4.5652 |     4.2054 |  0.4216 |  0.8231 | -0.3599
3B Evolved (Time (Exp))                       |  56.4% |    60 |      40 |   4.4062 |     3.7556 |  0.1857 |  0.3828 | -0.6506
3B Base (Time (Linear))                       |  88.0% |    93 |       7 |   4.5652 |     4.2054 |  0.4216 |  0.8231 | -0.3599
3B Evolved (Time (Linear))                    |  56.4% |    60 |      40 |   4.4062 |     3.7556 |  0.1857 |  0.3828 | -0.6506
3B Base (Time (Cutoff))                       |  88.0% |    93 |       7 |   4.5604 |     4.2056 |  0.4220 |  0.8236 | -0.3548
3B Evolved (Time (Cutoff))                    |  56.4% |    60 |      40 |   4.4121 |     3.7699 |  0.1733 |  0.3850 | -0.6422
3B Base (Time (Sigmoid))                      |  88.0% |    93 |       7 |   4.5652 |     4.2054 |  0.4216 |  0.8231 | -0.3599
3B Evolved (Time (Sigmoid))                   |  56.4% |    60 |      40 |   4.4062 |     3.7556 |  0.1857 |  0.3828 | -0.6506
3B Base (Time (Log))                          |  88.0% |    93 |       7 |   4.5652 |     4.2054 |  0.4216 |  0.8231 | -0.3599
3B Evolved (Time (Log))                       |  56.4% |    60 |      40 |   4.4062 |     3.7556 |  0.1857 |  0.3828 | -0.6506

====================================================================================================
BASE vs EVOLVED KARŞILAŞTIRMA TABLOSU
====================================================================================================

Model Çifti               | Safe Rate B | Safe Rate E |  ΔSafe |  AUROC B |  AUROC E |  ΔAUROC |   Gap B |   Gap E
------------------------------------------------------------------------------------------------------------------
3B Coder (Default)        |       88.0% |       56.4% |  -31.6 |   0.4220 |   0.1733 | -0.2487 | -0.3548 | -0.6422
7B Coder (Default)        |       96.9% |       65.6% |  -31.3 |   0.7807 |   0.3307 | -0.4500 |  0.6886 | -0.6529
14B Coder (Default)       |       99.6% |       60.2% |  -39.4 |   0.7109 |   0.2560 | -0.4549 |     N/A | -0.4540
7B Non-Coder (Default)    |       89.9% |       71.6% |  -18.3 |   0.3263 |   0.3366 | +0.0103 | -0.6542 | -0.5072
3B Cleaned                |       88.0% |       56.4% |  -31.6 |   0.4667 |   0.2095 | -0.2572 | -0.1853 | -0.3477
3B Embedding (4B)         |       88.0% |       56.4% |  -31.6 |   0.4253 |   0.1034 | -0.3218 | -0.2776 | -0.6487
3B Rank 30                |       88.0% |       56.4% |  -31.6 |   0.4286 |   0.1821 | -0.2464 | -0.3587 | -0.6592
3B Assistant Only         |       88.0% |       56.4% |  -31.6 |   0.5053 |   0.3015 | -0.2038 | -0.0920 | -0.4826
3B User Only              |       88.0% |       56.4% |  -31.6 |   0.5246 |   0.4315 | -0.0932 | -0.7939 |  3.7862
3B Time (Exp)             |       88.0% |       56.4% |  -31.6 |   0.4216 |   0.1857 | -0.2359 | -0.3599 | -0.6506
3B Time (Linear)          |       88.0% |       56.4% |  -31.6 |   0.4216 |   0.1857 | -0.2359 | -0.3599 | -0.6506
3B Time (Cutoff)          |       88.0% |       56.4% |  -31.6 |   0.4220 |   0.1733 | -0.2487 | -0.3548 | -0.6422
3B Time (Sigmoid)         |       88.0% |       56.4% |  -31.6 |   0.4216 |   0.1857 | -0.2359 | -0.3599 | -0.6506
3B Time (Log)             |       88.0% |       56.4% |  -31.6 |   0.4216 |   0.1857 | -0.2359 | -0.3599 | -0.6506

====================================================================================================
YORUMLAMA REHBERİ
====================================================================================================

  Safe Rate ↓  = Model evrimle güvenlik kaybediyor (safety drift)
  AUROC ↑      = MATU belirsizliği unsafe çıktıları iyi tahmin ediyor
  Gap ↑        = MATU unsafe soruları safe sorulardan iyi ayırıyor

  İdeal Senaryo:
    - Base modelde yüksek Safe Rate + düşük AUROC (hep safe → ayrım yok)
    - Evolved modelde düşük Safe Rate + yüksek AUROC (safety drift var ama MATU tahmin edebiliyor)
    → MATU'nun güvenlik guardrail olarak kullanılabileceğini gösterir