# Xiaoxingkai Transfer Summary

- Target lake: 【Li】Lake Xiaoxingkai
- Training lakes: 【Huo】Lake Hnagzhang | 【Huo】Lake Ulansu | 【North Temperate Lakes LTER】Crystal Lake (NTL) | 【North Temperate Lakes LTER】Bog 27-2 (Crystal Bog) | 【North Temperate Lakes LTER】Sparkling Lake | 【North Temperate Lakes LTER】Big Muskellunge Lake | 【North Temperate Lakes LTER】Allequash Lake | 【North Temperate Lakes LTER】Bog 12-15 (Trout Bog) | 【North Temperate Lakes LTER】Trout Lake | 【IISD Experimental】Lake 239 | 【IISD Experimental】Lake 302
- Validation RMSE: 0.3118
- Test RMSE: 0.3499

## Interpretation

- Xiaoxingkai rows before the cutoff are split into train and validation by time order.
- Xiaoxingkai rows from the cutoff onward are reserved for test only.
- Source lakes still use their own temporal train/val split.