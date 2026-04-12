# Unsupervised Learning Notes — Epic 4

CS 6140 · Neural Beings · April 2026

---

## Objective

Apply clustering and dimensionality reduction to the Met Museum's preprocessed feature matrix to discover latent groupings and evaluate alignment with curatorial departments and art historical periods — without using any labels during training.

---

## Pipeline Summary

1. **Standardization** — StandardScaler on the full 214-feature matrix. Necessary because frequency-encoded columns (Culture, Artist Nationality) range into the tens of thousands while TF-IDF and boolean features sit between 0 and 1. Without this, distance-based methods would be dominated by Culture and Nationality alone.

2. **Dimensionality Reduction** — TruncatedSVD to 50 components (59% variance retained). TruncatedSVD was chosen over PCA because PCA centers the data, destroying the sparsity structure of TF-IDF columns. 100 components were needed to reach 80% variance — the data is genuinely high-dimensional with no clean low-rank structure, typical for TF-IDF-heavy feature matrices.

3. **UMAP Visualization** — 2D projection of a 30k stratified subsample, colored by department. Run on a subsample because UMAP's computation time scales poorly with n; 30k is sufficient to reveal the structure without changing the visual pattern.

4. **K-Means Clustering** — Run on the full 388k training set (SVD-reduced). Tested k = {5, 6, ..., 30}. Evaluated at k = 6, 19, 25, 30.

5. **Hierarchical Clustering** — Agglomerative (Ward linkage) on a 15k stratified subsample. Subsampled because the algorithm requires an O(n²) distance matrix — 388k rows would need ~1.2 TB of RAM, infeasible regardless of compute resources.

6. **Evaluation** — Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) against department labels. Cross-tabulation heatmaps for interpretability.

7. **Test Set Validation** — Test artworks assigned to nearest training cluster center. Scores compared to training set to verify generalization.

---

## Dimensionality Reduction — TruncatedSVD

| Components | Cumulative Variance |
|---|---|
| 10 | 23.8% |
| 25 | 41.4% |
| 50 | 58.9% |
| 75 | 71.1% |
| 100 | 81.7% |

**Decision:** 50 components retained. The tail components increasingly capture TF-IDF noise rather than structure — including them would hurt clustering more than help. Can be revisited if clustering results are weak.

**Top component interpretation:**
- Component 1 — dominated by Period TF-IDF features. Separates objects with period information from those without (81% of Period is null).
- Component 2 — temporal/provenance axis. `object_age` and `Object End Date` load in opposite directions, alongside `Artist Nationality` and `Artist Role`.
- Components 3, 5 — more Period TF-IDF, likely distinguishing specific period labels (Chinese dynasties vs Egyptian kingdoms).
- Component 4 — Medium TF-IDF. Separates objects by material composition.

---

## K-Means Clustering

### Cluster Selection

| k | Inertia | Silhouette |
|---|---|---|
| 6 | 38,260,203 | 0.2436 |
| 19 | 24,203,442 | 0.1972 |
| 25 | 19,666,352 | 0.2360 |
| 30 | 16,937,005 | 0.2620 |

No clear elbow in the inertia curve. Silhouette scores are low overall (0.12–0.26), confirming that the data lacks crisp cluster boundaries. Spike at k=6 suggests ~6 broad natural groupings. Gradual climb from k=19 onward.

### Evaluation Against Department Labels

| k | ARI | NMI |
|---|---|---|
| 6 | 0.1689 | 0.3797 |
| 19 | 0.2797 | 0.5309 |
| 25 | 0.2484 | 0.5108 |
| 30 | 0.2603 | 0.5026 |

k=19 achieves the best alignment with departments (ARI=0.28, NMI=0.53).

### Key Findings — k=19

- Several clusters are nearly 100% pure — Asian Art, Photographs, and Egyptian Art are cleanly isolated.
- Drawings & Prints (35% of data) gets split across 4+ clusters — the algorithm sees sub-groups within it rather than one department.
- Messiest clusters mix European Sculpture, Costume Institute, Arms and Armor, and other Western departments.
- Small departments (Medieval Art, The Cloisters, Robert Lehman Collection, Islamic Art, The Libraries) get absorbed into larger mixed clusters rather than forming their own.

### Key Findings — k=6

- Asian Art splits into 3 pure sub-clusters — strong internal sub-structure (likely Chinese, Japanese, South Asian).
- An "Ancient World" macro-group emerges: Egyptian Art (41%) + Greek and Roman Art (48%) + Ancient Near Eastern Art (7%).
- Photographs + Drawings & Prints form a media-based cluster.
- Most Western/modern departments collapse into one large mixed cluster (73% of data).

This partially aligns with the 6 macro-groups used in the supervised hierarchical classifier (Ancient, Asian, European, American, Modern, Specialized). The unsupervised approach independently discovers the Ancient and Asian groupings but cannot separate European, American, and Modern departments without labels.

---

## Hierarchical Clustering

Run on 15k stratified subsample with Ward linkage.

### Evaluation Against Department Labels (k=19)

| Method | ARI | NMI |
|---|---|---|
| K-Means (388k) | 0.2797 | 0.5309 |
| Hierarchical (15k) | 0.0424 | 0.4064 |

Hierarchical clustering produces weaker quantitative alignment. Ward linkage makes different splitting decisions than K-Means — early merge mistakes propagate through the tree. The smaller subsample also contributes to less stable clusters.

### Key Findings

- Same broad patterns as K-Means: Asian Art splits into pure clusters, Photographs forms clean clusters, Drawings & Prints dominates multiple clusters.
- Cluster 5 merges Ancient Near Eastern Art (65%) with Asian Art (18%) and Greek/Roman (18%) — a different ancient-world grouping than K-Means produced.
- Cluster 17 merges Drawings & Prints (41%) with The American Wing (56%) — a grouping K-Means didn't find.
- The dendrogram's primary value is showing merge order, not competing with K-Means on accuracy.

### k=6 Comparison

Both algorithms agree on macro-structure:
- Asian Art is distinct and internally structured.
- Ancient departments (Egyptian, Greek/Roman) group together.
- Most Western/modern departments collapse into a single mixed cluster.

Specific sub-groupings differ at the edges, but consistent macro-structure across two independent algorithms is a stronger finding than either alone.

---

## Feature Group Ablation

K-Means (k=19) run separately on each feature group to identify what drives clustering.

| Feature Group | ARI | NMI |
|---|---|---|
| SVD-reduced (50 components) | 0.2797 | 0.5309 |
| Dense (14 features) | 0.2207 | 0.4699 |
| Period TF-IDF (50 features) | 0.1086 | 0.3072 |
| All raw features (214) | 0.0293 | 0.4078 |
| Tags TF-IDF (50 features) | -0.0285 | 0.0837 |
| Medium TF-IDF (100 features) | -0.0494 | 0.2406 |

### Key Findings

- **Dense features alone (14 columns) nearly match the full SVD-reduced result.** Structured fields like `Is Public Domain`, `object_age`, `Culture`, and `Classification` carry most of the clustering signal. This aligns with the supervised SHAP finding that `Is Public Domain` and `Artist Nationality` were the top two predictive features.
- **All 214 raw features perform poorly (ARI=0.03).** The 200 TF-IDF columns add noise that drowns out the dense feature signal — a direct demonstration of the curse of dimensionality.
- **Medium and Tags TF-IDF have negative ARI.** These features describe what objects are made of and what they depict, which cuts across departments. Oil on canvas appears in European Paintings, The American Wing, and Modern Art.
- **Period TF-IDF is the strongest text group** but still far weaker than dense features. Period labels like "New Kingdom" are department-specific, but 81% null coverage limits their reach.
- **SVD reduction is critical.** It extracts the useful signal from TF-IDF while filtering noise, outperforming both raw features and dense-only.

**Conclusion:** The Met's departmental structure is primarily encoded in structured metadata (era, provenance, object type) rather than in free-text descriptions of materials or subject tags.

---

## Art Historical Periods Analysis

The Period column is 81% null, but the ~19% of training records with period labels (~72,832 records, 1,687 unique periods) allow us to check whether objects from the same historical period cluster together.

### K-Means (k=19) — Top 15 Periods

The clusters map almost perfectly onto art-historical civilizations:

| Cluster | Civilization | Periods |
|---|---|---|
| 8 | Egyptian | New Kingdom, Middle Kingdom, New Kingdom Ramesside, New Kingdom Amarna, Third Intermediate Period, Late Period — all at 100% |
| 11 | Greek/Roman | Archaic, Classical, Archaic/Classical, Hellenistic, Early Imperial, Late Archaic — all at 99–100% |
| 6 | Chinese | Qing dynasty — 100% |
| 15 | Japanese (Edo) | Edo period — 100% |
| 7 | Japanese (transitional) | Edo or Meiji — 100% |

The algorithm does not distinguish sub-periods within a civilization (New Kingdom vs Middle Kingdom both go to Cluster 8), but it cleanly separates civilizations from each other. This confirms that the Met's metadata carries strong art-historical signal at the civilization level.

---

## Test Set Validation

| Set | ARI | NMI |
|---|---|---|
| Train | 0.2797 | 0.5309 |
| Test | 0.2777 | 0.5319 |

Scores differ by less than 0.01. The test set cross-tabulation is virtually identical to the training set — same pure clusters, same mixed clusters, same percentages. The clustering structure is stable and generalizable.

---

## Key Takeaways

1. **K-Means on SVD-reduced features is the best unsupervised approach** for this dataset — ARI=0.28, NMI=0.53 at k=19.

2. **Some departments are cleanly separable from metadata alone** — Egyptian Art, Asian Art, Photographs, and Greek/Roman Art form distinct clusters. These departments have unique combinations of era, materials, and provenance that make them metadata-distinctive.

3. **Other departments are indistinguishable in metadata space** — Medieval Art, The Cloisters, Robert Lehman Collection, and the European departments overlap heavily. Their separation depends on curatorial provenance decisions that aren't captured in the metadata. This mirrors the supervised finding that The Cloisters (F1=0.85) and Medieval Art (F1=0.93) were the hardest pair for classification.

4. **The algorithm independently discovers art-historical civilizations** — without seeing Period labels, K-Means organizes Egyptian, Greek/Roman, Chinese, and Japanese objects into nearly pure clusters. The departmental structure curators built is genuinely reflected in the metadata.

5. **Structured features dominate over text features** — 14 dense columns (booleans, dates, encoded categoricals) carry more clustering signal than 200 TF-IDF columns. SVD reduction is necessary to extract the useful signal from TF-IDF while suppressing noise. This aligns with the supervised SHAP analysis showing `Is Public Domain` and `Artist Nationality` as the top predictive features.

6. **The unsupervised macro-structure partially matches the supervised hierarchy** — K-Means at k=6 discovers Ancient and Asian groupings independently, validating the domain-knowledge grouping used in the hierarchical XGBoost classifier.

---

## Output Files

### Figures
| File | Description |
|---|---|
| `outputs/figures/svd_variance_explained.png` | Cumulative and per-component variance plots |
| `outputs/figures/umap_by_department.png` | UMAP 2D projection colored by department (30k subsample) |
| `outputs/figures/kmeans_elbow_silhouette.png` | Elbow plot and silhouette scores for k=5–30 |
| `outputs/figures/kmeans_clusters_umap.png` | K-Means cluster assignments on UMAP for k=6, 19, 25, 30 |
| `outputs/figures/kmeans_19_crosstab_heatmap.png` | Department composition per cluster — K-Means k=19 |
| `outputs/figures/kmeans_6_crosstab_heatmap.png` | Department composition per cluster — K-Means k=6 |
| `outputs/figures/hierarchical_dendrogram.png` | Ward linkage dendrogram (15k subsample) |
| `outputs/figures/hierarchical_19_crosstab_heatmap.png` | Department composition per cluster — Hierarchical k=19 |
| `outputs/figures/hierarchical_6_crosstab_heatmap.png` | Department composition per cluster — Hierarchical k=6 |
| `outputs/figures/ablation_feature_groups.png` | ARI/NMI by feature group |
| `outputs/figures/kmeans_19_period_crosstab.png` | Period vs cluster alignment — K-Means k=19 |
| `outputs/figures/kmeans_19_test_crosstab_heatmap.png` | Test set department composition — K-Means k=19 |