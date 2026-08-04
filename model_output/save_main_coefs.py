"""Extract & save the main-spec distance coefficient table from the fitted model."""
import joblib
import pandas as pd
from common import DIST_FEATS

OUT = "model_output"
clf = joblib.load(f"{OUT}/model_main.joblib")
cols = joblib.load(f"{OUT}/feature_cols_main.joblib")
coef_series = pd.Series(clf.coef_[0], index=cols)
dist_coef = coef_series.loc[DIST_FEATS]
out = pd.DataFrame({"coef": dist_coef, "sign": dist_coef.apply(lambda x: "negative" if x < 0 else "positive")})
out = out.reindex(dist_coef.abs().sort_values(ascending=False).index)
out.to_csv(f"{OUT}/distance_coefficients_main_only.csv")
print(out)
