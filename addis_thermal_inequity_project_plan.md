# Thermal Inequity Mapping: Quantifying the Micro-Climate Penalty of Addis Ababa's Informal Settlements via Satellite Embeddings (2017–2024)

**Status:** Locked design — ready to execute
**Scope:** Addis Ababa administrative boundary (~540 km²)
**Compute:** Laptop GPU (this pipeline is intentionally GPU-light — see note in Section 4)

---

## 1. Research Question

Do Addis Ababa's informal settlements bear a quantifiable, worsening temperature penalty relative to formal urban fabric, and can a satellite foundation model (AlphaEarth) detect where that penalty is emerging as settlements expand — without hand-crafted spectral indices?

**Why this is novel (verified via literature check):** at least six 2024–2026 studies quantify Addis Ababa's urban heat island, but every one relies on NDVI/NDBI + linear or geographically-weighted regression. The closest any comes to an equity framing splits results by "urban morphology type" (apartments/villas/mud houses) using SPSS regression. None uses learned satellite representations, and none is explicitly framed around informal-settlement inequity. That's the gap.

**Honest scope correction from the original pitch:** the original framing talked about "isolating individual corrugated iron roofs" at high resolution. This isn't physically recoverable — Landsat thermal is 100m native/30m resampled, and individual dwellings (20–80m²) are smaller than even a single 10m AlphaEarth pixel. The real, defensible claim is **block/neighborhood-scale fabric classification correlated with block-scale thermal load** — still a genuine methodological improvement over hand-crafted-index regression, just not per-roof.

---

## 2. Data Sources

| Layer | Source | Resolution | Years | Role |
|---|---|---|---|---|
| Land representation | AlphaEarth Satellite Embedding V1 Annual (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` in Earth Engine) | 10m, 64-dim | 2017, 2024 | Classifier input |
| Thermal | Landsat 8/9 Collection 2 Level-2 Surface Temperature band | 30m (100m native, resampled) | 2017, 2024 | Outcome variable |
| Visualization only | Sentinel-2 L2A RGB (B4/B3/B2) | 10m | 2017, 2024 | Demo visuals, NOT used in training |
| Confound control | NDVI from Sentinel-2, elevation from SRTM (`USGS/SRTMGL1_003`) | 10–30m | static/2017/2024 | Regression covariates |
| Labels | Manually digitized polygons | — | 2024 basemap | Ground truth (see Section 3) |

**Seasonal note:** composite Landsat LST from dry-season months (roughly December–February) in both years to minimize cloud/rain contamination from Ethiopia's two rainy seasons.

**Access method:** pull everything through the Earth Engine Python API (`earthengine-api` / `geemap`), clip to the Addis Ababa administrative boundary, export as local GeoTIFFs. At 10m over 540km², the full embedding stack is well under 1GB — trivial for a laptop.

---

## 3. Labels

No ready-made informal-settlement shapefile exists publicly for Addis. Instead:

- Hand-digitize ~100–150 polygons in QGIS over free high-resolution basemap imagery (Google/Esri/Bing), split into two classes: **informal settlement** vs. **everything else**.
- Guide polygon placement using literature-identified deprivation hotspots: a 2023 spatial deprivation-index study found high-deprivation clustering in the peri-urban/suburban areas of **Bole** and **Akaki Kaliti**, plus inner-city slum pockets in **Chirkos**, **Arada**, and **Lideta**, versus low-deprivation formal fabric in **Gulele**, **Kolfe-Keranyo**, and parts of **Bole**/**Yeka**.
- **Label only against 2024 imagery** (most reliable, current basemap). Do not attempt to re-digitize historical 2017 boundaries — see Section 5 for why.
- Sample ~30–50 pixel points per polygon for training (thousands of pixel-level samples total from 100–150 polygons — plenty for a 64-dim linear/shallow classifier).
- **Split train/val/test by polygon, not by pixel.** Pixels within the same polygon are spatially correlated; random pixel-level splitting will leak and inflate accuracy.

---

## 4. Model

- **Per-pixel classifier**, not spatial/patch-based: a small MLP (and an XGBoost baseline for comparison) taking the 64-dim AlphaEarth embedding vector as input, predicting informal-vs-other. This mirrors how AlphaEarth's own paper evaluates downstream tasks (linear/shallow probes on embeddings).
- **GPU-honesty note:** this pipeline is intentionally light on compute — the whole point of embedding-based methods is to skip heavy training. Training the MLP on tens of thousands of 64-dim samples takes seconds to low minutes even on CPU. That's expected and fine; it's a selling point of the approach, not a shortfall. If you want a GPU-heavier extension later, a patch-based CNN (5×5/9×9 window) is a natural ablation to add.

---

## 5. Multi-Year Design (Why Train Once, Apply Twice)

Rather than digitizing separate labels for 2017 and 2024 (which would introduce label-consistency noise — did you draw the same boundary both times?), the design is:

1. Train the classifier **once**, only on 2024 labels (the year you can verify against current imagery).
2. Apply that **frozen, unchanged classifier** to both the 2017 and 2024 embedding rasters.
3. Any difference in classification output between the two years reflects real land-cover change (settlement growth), not labeling drift, because the decision function never changed.
4. Pixels that flip from "other" → "informal" between 2017 and 2024 = settlement expansion zones. This gives you a genuine change-detection map for free.

---

## 6. Statistical Analysis

- For each year, compare Landsat LST distributions between the two predicted classes using a non-parametric test (Mann–Whitney U — LST distributions are unlikely to be normal), not just a raw mean difference.
- Fit a regression of LST on {predicted class, NDVI, elevation} to isolate the informal-settlement effect **independent of** vegetation and elevation — this is the concrete methodological upgrade over the existing Addis literature, which reports NDVI/NDBI correlations without an explicit settlement-type term.
- Compare the 2017 effect size to the 2024 effect size (with confidence intervals) to state whether the gap widened, narrowed, or held steady — this is your headline result.

---

## 7. Deliverables

1. **Code**: Earth Engine export scripts → QGIS labeling files → training/eval notebook → statistics notebook → visualization/video-export script.
2. **Short paper** (markdown → PDF), structured as:
   - Abstract
   - Introduction / Motivation
   - Related Work (existing Addis NDVI/NDBI/LST studies; AlphaEarth Foundations)
   - Data & Labels
   - Method
   - Results (classification accuracy, LST-by-class comparison, 2017→2024 trend)
   - Discussion & Limitations (see Section 8 — state these explicitly, don't bury them)
   - Conclusion
3. **Video-ready visuals**: side-by-side Sentinel-2 RGB / classification map / LST heatmap, for both years, plus a settlement-growth overlay and a distribution plot (violin/box) of LST by class and year.

---

## 8. Limitations to State Explicitly in the Paper

Stating these upfront makes the paper more credible, not less:

- **Resolution ceiling**: 30m thermal vs. 10m land classification is a real mismatch; no claim about individual-building temperatures is supported.
- **Labels are self-digitized**, not field-verified against a ground survey — there's annotator judgment/uncertainty baked into the "informal" boundary itself.
- **Correlational, not causal**: the analysis shows association between fabric type and LST, controlling for two confounders — it doesn't establish mechanism.
- **Single city**: findings don't automatically generalize to other Ethiopian or East African cities without further work.
- **LST ≠ lived heat exposure**: land surface temperature is not the same as air temperature people actually experience; this is a proxy, worth naming as one.

---

## 9. Estimated Timeline (Laptop GPU)

| Task | Time |
|---|---|
| Earth Engine data pull + export | <1 hour |
| Manual labeling in QGIS | 3–6 hours |
| Model training + evaluation | Minutes |
| Statistical analysis | 1–2 hours |
| Visualization/video assembly | 3–5 hours |
| Paper writing | 3–5 days |
| **Total** | **~1.5–2.5 weeks**, comfortably within your window |

---

## References (for your Related Work section)

- AlphaEarth Foundations paper: https://arxiv.org/pdf/2507.22291
- AlphaEarth Satellite Embedding dataset (Earth Engine): https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- Addis Ababa UHI/LULC 1990–2024 study: https://www.sciencedirect.com/science/article/abs/pii/S221067072500887X
- Addis Ababa LST spatial modeling: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10942972/
- Addis Ababa UMT-based microclimate study: https://link.springer.com/article/10.1007/s44274-024-00105-6
- Spatial deprivation pattern study (Addis hotspots): https://www.mdpi.com/2071-1050/15/3/1934
- Deprived-area ML mapping methodology (Accra/Lagos/Nairobi): https://www.mdpi.com/2413-8851/7/4/116

---

## Next Step

Ready to start executing whenever you are — the first concrete artifact would be the Earth Engine Python script that pulls and exports the 2017 and 2024 AlphaEarth + Landsat + Sentinel-2 layers clipped to Addis Ababa.
