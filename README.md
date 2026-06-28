# Perovskite Stability Screening
Predicting the thermodynamic stability of halide perovskite compositions (energy above hull < 0.1eV/atom)
directly from chemical formula, using Materials Project data and composition-based machine learning.

## Motivation
Halide perovskites are widely studied for solar cell and optoelectronic
applications. Understanding their stability is very important before usage in a real-world context.
Before investing in expensive DFT calculations or experimental synthesis when 
finding new candidates, it's useful to screen candidates by predicted
stability using only their chemical formula as a low-cost, high speed prioritization method for testing novel materials.

## Approach
- **Data**: Queried from the Materials Project API, filtered to compositions containing exactly one B-site metal (Pb, Sn, or Ge)
  and at least one halide, with an A-site element present and ≤4 total elements, band gap 0.5–3.0 eV.
  This admits some compositions beyond simple ABX₃ stoichiometry including mixed-anion and more complex substituted structures
  (e.g., compositions containing oxygen alongside halides).
- **Features**: Compositional descriptors from 'matminer' MAGPIE featurizer (elemental statistics: electronegativity, atomic weight, valence counts, etc.).
  We also included spacegroup number as well as goldschmidt tolerance factor initially (V3-V6).
  The point of both of these was to include geometric structural information beyond just the composition.
  Ultimately, they had to be removed: spacegroup as its unknowable structural information for novel, untested, compositions
  and the Goldschmidt tolerance factor due to restriction to ABX3 structure.
- **Models compared**: Random Forest, XGBoost, Ridge Regression, and
  Gaussian Process Regression, evaluated via 5-fold cross-validation.
  XGBoost was selected as the best performer.
- **Target**: `energy_above_hull` (eV/atom) which is the distance from the
  thermodynamic stability hull. Lower values indicate more stable,
  more likely synthesizable compositions. A common threshold for stability is set at 0.1eV/atom.

## Key methodological findings
Two issues surfaced during development:
1. **Feature leakage via spacegroup number.** Earlier versions (V4-V6) included the
   DFT-measured spacegroup number as a feature. This is structural
   information that is unknowable for an uncharacterized
   candidate composition. We confirmed via
   cross-validation in V6 that excluding it did not meaningfully change model
   performance or which model ranked best, then removed it. A different,
   composition-derived MAGPIE statistic (built from elements' reference
   space groups, not the compound's actual structure) was retained, since it
   is legitimately computable from formula alone. Interestingly, these element-specific 
   spacegroup numbers are among the most impactful features for the model (in top 15).
2. **Polymorph degeneracy.** Many compositions have multiple known polymorphs in Materials Project, each with a   
   different hull energy. Since our features are purely compositional,
   different polymorphs of the same formula are indistinguishable to the
   model, meaning duplicate formulas with different target values would
   otherwise feed the model contradictory training signal. We resolve this
   by selecting the lowest-energy polymorph per
   composition, which is also the physically meaningful target for
   stability specifically.
## Results
XGBoost achieved a cross-validated R² of ~0.61 -0.64 and a MAE (mean absolute error) of
~0.047 eV/atom on a showcase set of candidate compositions. This error is not evenly distributed. In V7, the one mixed halide-oxygen candidate in our showcase 
set (NiSn(ClO)₆) had substantially higher error than the standard ABX₃ candidates. However, this is based on a single example rather
than a systematic comparison across the dataset.
Training R² (~0.999) is also notably higher than test R² (~0.68), indicating overfitting given the feature-to-sample ratio (~130 features, ~290 samples).
While XGBoost performed the best in terms of test R^2 magnitude and standard deviation, it is
worth noting that it possessed the highest train R^2 of the model candidates. 
## Next Steps
1) Expand dataset to include perovskite structures like Double Perovskites (A₂BB'X₆) etc.
   The current Dataset is quite small, and icluding more possible structures might be helpful for model generalization. 
2) Reintroduce tolerance factor metrics. We removed the Goldschmidt tolerance factor because
   it requires ABX3 structure and we wanted to limit our dataset size less.
   There are other tolerance factor metrics that generalize to different structures and we hope to implement those next.

## File nomenclature and usage
The notebooks, detailed V1-V7, provide my progression of exploration through the project.
The most recent notebook version, V7 is the most updated version, but you can trial other versions to see the progression.
I kept all versions there to showcase the project development. 

## Feedback
This project reflects my ongoing learning in materials informatics and applied ML.
I'd genuinely welcome technical feedback, whether on the modeling choices, the domain framing, or anything I've missed.
   
## Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib matminer pymatgen mp-api python-dotenv 
```
A free Materials Project API key (only needed if re-querying live data rather than using the included CSV snapshot). Set as `MP_API_KEY` in a `.env` file in the project root.
