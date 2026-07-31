====================================================================================================
HARMBENCH MATU GÜVENLIK BELİRSİZLİK ANALİZİ
====================================================================================================

--- Qwen2.5-Coder-3B-Instruct (Base) (harmbench_3b_base) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 78.7%
  Safe Behavior (majority): 88/100
  Unsafe Behavior         : 12/100

  Genel Ortalama Unc.     : 4.7000
  Genel Median Unc.       : 4.7677
  SAFE  Sorular Unc.      : 4.6921
  UNSAFE Sorular Unc.     : 4.7576
  GAP (Unsafe - Safe)     : 0.0655  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.6995
  AUROC (majority_incorr) : 0.5047
  AUARC                   : 0.8360

--- AZR-Coder-3b (Evolved) (harmbench_3b_evolved) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 40.0%
  Safe Behavior (majority): 41/100
  Unsafe Behavior         : 59/100

  Genel Ortalama Unc.     : 4.3551
  Genel Median Unc.       : 4.2765
  SAFE  Sorular Unc.      : 4.3281
  UNSAFE Sorular Unc.     : 4.3739
  GAP (Unsafe - Safe)     : 0.0458  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.4547
  AUROC (majority_incorr) : 0.5316
  AUARC                   : 0.4044

--- Qwen2.5-Coder-7B-Instruct (Base) (harmbench_7b_base) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 85.5%
  Safe Behavior (majority): 89/100
  Unsafe Behavior         : 11/100

  Genel Ortalama Unc.     : 3.4682
  Genel Median Unc.       : 3.7848
  SAFE  Sorular Unc.      : 3.3344
  UNSAFE Sorular Unc.     : 4.5503
  GAP (Unsafe - Safe)     : 1.2158  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.8460
  AUROC (majority_incorr) : 0.7763
  AUARC                   : 0.9404

--- AZR-Coder-7b (Evolved) (harmbench_7b_evolved) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 49.2%
  Safe Behavior (majority): 50/100
  Unsafe Behavior         : 50/100

  Genel Ortalama Unc.     : 4.1571
  Genel Median Unc.       : 3.9986
  SAFE  Sorular Unc.      : 4.0297
  UNSAFE Sorular Unc.     : 4.2845
  GAP (Unsafe - Safe)     : 0.2548  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.5120
  AUROC (majority_incorr) : 0.5936
  AUARC                   : 0.5645

--- Qwen2.5-Coder-14B-Instruct (Base) (harmbench_14b_base) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 84.8%
  Safe Behavior (majority): 91/100
  Unsafe Behavior         : 9/100

  Genel Ortalama Unc.     : 3.5541
  Genel Median Unc.       : 3.6161
  SAFE  Sorular Unc.      : 3.5043
  UNSAFE Sorular Unc.     : 4.0573
  GAP (Unsafe - Safe)     : 0.5530  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.7027
  AUROC (majority_incorr) : 0.6691
  AUARC                   : 0.9103

--- AZR-Coder-14b (Evolved) (harmbench_14b_evolved) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 46.9%
  Safe Behavior (majority): 47/100
  Unsafe Behavior         : 53/100

  Genel Ortalama Unc.     : 3.9061
  Genel Median Unc.       : 3.8292
  SAFE  Sorular Unc.      : 3.7716
  UNSAFE Sorular Unc.     : 4.0254
  GAP (Unsafe - Safe)     : 0.2538  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.5984
  AUROC (majority_incorr) : 0.6315
  AUARC                   : 0.5493

--- Qwen2.5-7B-Instruct (Base, Non-Coder) (harmbench_7b_base_noncoder) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 62.6%
  Safe Behavior (majority): 61/100
  Unsafe Behavior         : 39/100

  Genel Ortalama Unc.     : 3.6826
  Genel Median Unc.       : 3.6235
  SAFE  Sorular Unc.      : 3.4571
  UNSAFE Sorular Unc.     : 4.0352
  GAP (Unsafe - Safe)     : 0.5781  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.7768
  AUROC (majority_incorr) : 0.7625
  AUARC                   : 0.8009

--- AZR-Base-7b (Evolved, Non-Coder) (harmbench_7b_evolved_noncoder) ---
  Toplam Behavior         : 100
  Safe Rate (per-run)     : 50.5%
  Safe Behavior (majority): 53/100
  Unsafe Behavior         : 47/100

  Genel Ortalama Unc.     : 4.1029
  Genel Median Unc.       : 4.0618
  SAFE  Sorular Unc.      : 4.1537
  UNSAFE Sorular Unc.     : 4.0456
  GAP (Unsafe - Safe)     : -0.1081  <-- Büyük = İyi

  AUROC (any_incorrect)   : 0.3698
  AUROC (majority_incorr) : 0.4420
  AUARC                   : 0.4736

====================================================================================================
ÖZET TABLO
====================================================================================================

Deney                                         |  Safe% | #Safe | #Unsafe |   AUROC |   AUARC |     Gap
------------------------------------------------------------------------------------------------------
Qwen2.5-Coder-3B-Instruct (Base)              |  78.7% |    88 |      12 |  0.6995 |  0.8360 |  0.0655
AZR-Coder-3b (Evolved)                        |  40.0% |    41 |      59 |  0.4547 |  0.4044 |  0.0458
Qwen2.5-Coder-7B-Instruct (Base)              |  85.5% |    89 |      11 |  0.8460 |  0.9404 |  1.2158
AZR-Coder-7b (Evolved)                        |  49.2% |    50 |      50 |  0.5120 |  0.5645 |  0.2548
Qwen2.5-Coder-14B-Instruct (Base)             |  84.8% |    91 |       9 |  0.7027 |  0.9103 |  0.5530
AZR-Coder-14b (Evolved)                       |  46.9% |    47 |      53 |  0.5984 |  0.5493 |  0.2538
Qwen2.5-7B-Instruct (Base, Non-Coder)         |  62.6% |    61 |      39 |  0.7768 |  0.8009 |  0.5781
AZR-Base-7b (Evolved, Non-Coder)              |  50.5% |    53 |      47 |  0.3698 |  0.4736 | -0.1081
3B Base (Cleaned)                             |  78.7% |    88 |      12 |  0.6067 |  0.8059 | -0.1217
3B Evolved (Cleaned)                          |  40.0% |    41 |      59 |  0.5146 |  0.4690 |  0.3081
3B Base (Embedding (4B))                      |  78.7% |    88 |      12 |  0.6942 |  0.8424 |  0.0457
3B Evolved (Embedding (4B))                   |  40.0% |    41 |      59 |  0.4654 |  0.4361 |  0.2298
3B Base (Rank 30)                             |  78.7% |    88 |      12 |  0.5008 |  0.7752 | -0.2614
3B Evolved (Rank 30)                          |  40.0% |    41 |      59 |  0.5146 |  0.3791 |  0.1004
3B Base (Assistant Only)                      |  78.7% |    88 |      12 |  0.6966 |  0.8222 |  0.0203
3B Evolved (Assistant Only)                   |  40.0% |    41 |      59 |  0.4624 |  0.3936 | -0.0132
3B Base (User Only)                           |  78.7% |    88 |      12 |  0.5571 |  0.8086 | -0.5278
3B Evolved (User Only)                        |  40.0% |    41 |      59 |  0.7481 |  0.4031 | -3.1413
3B Base (Time (Exp))                          |  78.7% |    88 |      12 |  0.5021 |  0.7720 | -0.2631
3B Evolved (Time (Exp))                       |  40.0% |    41 |      59 |  0.5100 |  0.3781 |  0.0836
3B Base (Time (Linear))                       |  78.7% |    88 |      12 |  0.5021 |  0.7720 | -0.2631
3B Evolved (Time (Linear))                    |  40.0% |    41 |      59 |  0.5100 |  0.3781 |  0.0836
3B Base (Time (Cutoff))                       |  78.7% |    88 |      12 |  0.4959 |  0.7707 | -0.2672
3B Evolved (Time (Cutoff))                    |  40.0% |    41 |      59 |  0.4992 |  0.3731 |  0.0646
3B Base (Time (Sigmoid))                      |  78.7% |    88 |      12 |  0.5021 |  0.7720 | -0.2631
3B Evolved (Time (Sigmoid))                   |  40.0% |    41 |      59 |  0.5100 |  0.3781 |  0.0836
3B Base (Time (Log))                          |  78.7% |    88 |      12 |  0.5021 |  0.7720 | -0.2631
3B Evolved (Time (Log))                       |  40.0% |    41 |      59 |  0.5100 |  0.3781 |  0.0836

====================================================================================================
BASE vs EVOLVED KARŞILAŞTIRMA TABLOSU
====================================================================================================

Model Çifti               | Safe Rate B | Safe Rate E |  ΔSafe |  AUROC B |  AUROC E |  ΔAUROC |   Gap B |   Gap E
--------------------------------------------------------------------------------------------------------------------
3B Coder (Default)        |       78.7% |       40.0% |  -38.7 |   0.6995 |   0.4547 | -0.2448 |  0.0655 |  0.0458
7B Coder (Default)        |       85.5% |       49.2% |  -36.3 |   0.8460 |   0.5120 | -0.3340 |  1.2158 |  0.2548
14B Coder (Default)       |       84.8% |       46.9% |  -37.9 |   0.7027 |   0.5984 | -0.1042 |  0.5530 |  0.2538
7B Non-Coder (Default)    |       62.6% |       50.5% |  -12.1 |   0.7768 |   0.3698 | -0.4070 |  0.5781 | -0.1081
3B Cleaned                |       78.7% |       40.0% |  -38.7 |   0.6067 |   0.5146 | -0.0921 | -0.1217 |  0.3081
3B Embedding (4B)         |       78.7% |       40.0% |  -38.7 |   0.6942 |   0.4654 | -0.2287 |  0.0457 |  0.2298
3B Rank 30                |       78.7% |       40.0% |  -38.7 |   0.5008 |   0.5146 | +0.0138 | -0.2614 |  0.1004
3B Assistant Only         |       78.7% |       40.0% |  -38.7 |   0.6966 |   0.4624 | -0.2343 |  0.0203 | -0.0132
3B User Only              |       78.7% |       40.0% |  -38.7 |   0.5571 |   0.7481 | +0.1910 | -0.5278 | -3.1413
3B Time (Exp)             |       78.7% |       40.0% |  -38.7 |   0.5021 |   0.5100 | +0.0079 | -0.2631 |  0.0836
3B Time (Linear)          |       78.7% |       40.0% |  -38.7 |   0.5021 |   0.5100 | +0.0079 | -0.2631 |  0.0836
3B Time (Cutoff)          |       78.7% |       40.0% |  -38.7 |   0.4959 |   0.4992 | +0.0033 | -0.2672 |  0.0646
3B Time (Sigmoid)         |       78.7% |       40.0% |  -38.7 |   0.5021 |   0.5100 | +0.0079 | -0.2631 |  0.0836
3B Time (Log)             |       78.7% |       40.0% |  -38.7 |   0.5021 |   0.5100 | +0.0079 | -0.2631 |  0.0836

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
