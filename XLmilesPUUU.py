# -*- coding: utf-8 -*-
"""
XMiles Forecast – time-safe, leakage-safe (FINAL)
-------------------------------------------------
- 移除洩漏特徵、時間安全切分、單一 05:00 視窗（pay_time_local）
- 機率校準（isotonic）+ 全域 F1 門檻
- 7D/shift(1) 群組滾動特徵
- 分倉門檻：樣本量 + 品質(AUC) 收縮到全域，並有動態安全帶
- 量級對齊：用近 7 天真實比例作先驗，當天以分位數微調每倉門檻
- 7日回測：只用來**平滑 global_thr**，不覆蓋當天「量級對齊」後的門檻

輸出 (./outputs):
- Today_Forecast_Address_Summary.xlsx（Summary/Detail/Validation/Thresholds/Charts）
- Today_Forecast_Address_Summary.txt
"""

import inspect
import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy import sparse

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression   
import lightgbm as lgb     
from catboost import CatBoostClassifier
from scipy.interpolate import UnivariateSpline                          


# -----------------------------
# 0) MySQL 連線
# -----------------------------
DB_CFG = {
    'host': '',
    'port': 3306,
    'user': '',
    'password': '',
    'database': '',
    'charset': ''
}

def make_engine(cfg=DB_CFG):
    url = URL.create(
        "mysql+pymysql",
        username=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        query={"charset": cfg["charset"]}
    )
    return create_engine(url, pool_recycle=3600, pool_pre_ping=True, future=True)

def read_mysql_to_df(query: str, chunksize: int | None = None) -> pd.DataFrame:
    eng = make_engine()
    with eng.connect() as conn:
        sql = text(query)
        if chunksize:
            pieces = []
            for i, chk in enumerate(pd.read_sql_query(sql, conn, chunksize=chunksize)):
                print(f" - read chunk {i+1} (rows={len(chk)})")
                pieces.append(chk)
            return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        else:
            return pd.read_sql_query(sql, conn)

# ===========================================
# 1) MySQL 查詢
# ===========================================
QUERY_TMPL_ALL = r"""
SELECT * FROM `dwd_xlmiles_order_details2`
"""

# ===========================================
# 2) 特徵工程 + 標籤
# ===========================================

def clean_address(address):
    if pd.isna(address):
        return address
    s = str(address).lower()
    s = re.sub(r'\brd\b',  'road',   s)
    s = re.sub(r'\bave\b', 'avenue', s)
    s = re.sub(r'\bste\b', 'suite',  s)
    s = re.sub(r'\bst\b',  'street', s)
    s = re.sub(r'\bdr\b',  'drive',  s)
    s = re.sub(r'[^\w\s-]', '', s).replace('-', '')
    s = re.sub(r'\s+', ' ', s).strip()
    parts = s.split()
    if len(parts) > 2:
        s = ' '.join(parts[:3])
    return s

def _compute_deadline_17(pay_ts: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(pay_ts): return pd.NaT
    anchor_day = (pay_ts.normalize() - pd.Timedelta(days=1)) if pay_ts.hour < 5 else pay_ts.normalize()
    return anchor_day + pd.Timedelta(days=1, hours=17)

def add_window_and_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['pay_time_local']    = pd.to_datetime(out['pay_time_local'], errors='coerce')
    out['online_time_local'] = pd.to_datetime(out['online_time_local'], errors='coerce')
    out['deadline_17'] = out['pay_time_local'].apply(_compute_deadline_17)
    diff_min = (out['online_time_local'] - out['pay_time_local']).dt.total_seconds() / 60
    valid = (~out['online_time_local'].isna()) & (out['online_time_local'] >= out['pay_time_local'])
    out['online_minutes_local'] = np.where(valid, diff_min, np.nan).astype('float32')
    out['online_by_17'] = ((~out['online_time_local'].isna()) &
                           (out['online_time_local'] <= out['deadline_17'])).astype('int8')
    return out

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['pay_hour'] = out['pay_time_local'].dt.hour
    out['pay_weekday'] = out['pay_time_local'].dt.weekday
    out['is_weekend'] = out['pay_weekday'].isin([5,6]).astype('int8')
    out['hour_bin'] = pd.cut(out['pay_hour'], bins=[-1,5,11,17,23], labels=['凌晨','早班','午班','晚班'])
    anchor_day = np.where(out['pay_hour'] < 5,
                          (out['pay_time_local'] - pd.Timedelta(days=1)).dt.normalize(),
                          out['pay_time_local'].dt.normalize())
    anchor_day = pd.to_datetime(anchor_day)
    out['cutoff_1430'] = anchor_day + pd.Timedelta(hours=14, minutes=30)
    out['min_to_cutoff'] = (out['cutoff_1430'] - out['pay_time_local']).dt.total_seconds()/60
    out['min_to_deadline_from_pay'] = (out['deadline_17'] - out['pay_time_local']).dt.total_seconds()/60
    return out

def add_address_clean_and_hist_rate_17(df: pd.DataFrame, early_window_days: int = 2) -> pd.DataFrame:
    # 1) 用連續整數索引，避免之後 .loc 因為 DatetimeIndex 對不上而失敗
    out = df.copy().reset_index(drop=True)

    # 2) 清地址
    out['shipper_address1_clean'] = (
        out.get('shipper_address1', pd.Series(index=out.index))
          .apply(clean_address)
          .astype('string')
    )

    # 3) 視窗 & 臨時標籤
    out['deadline_17'] = pd.to_datetime(out['pay_time_local'], errors='coerce').apply(_compute_deadline_17)
    out['online_time_local'] = pd.to_datetime(out['online_time_local'], errors='coerce')
    lower_bound = out['deadline_17'] - pd.Timedelta(days=early_window_days)

    out['online_by_17_tmp'] = (
        (~out['online_time_local'].isna()) &
        (out['online_time_local'] <= out['deadline_17']) &
        (out['online_time_local'] >= lower_bound)
    ).astype('int8')

    # 4) 滾動 30 天歷史率（每個地址獨立，shift(1) 防洩漏）
    out['shipper_address1_historical_rate'] = np.nan

    mask = (~out['pay_time_local'].isna()) & (~out['shipper_address1_clean'].isna())
    calc = out.loc[mask].copy().sort_values(['shipper_address1_clean', 'pay_time_local'], kind='stable')

    for _, idx in calc.groupby('shipper_address1_clean', sort=False).groups.items():
        g = calc.loc[idx].sort_values('pay_time_local', kind='stable')
        row_ids = g.index.to_numpy()  
        g2 = g.set_index('pay_time_local')
        r = g2['online_by_17_tmp'].shift(1).rolling('30D').mean()

        out.loc[row_ids, 'shipper_address1_historical_rate'] = r.to_numpy(dtype='float32')

    out = out.drop(columns=['online_by_17_tmp'])

    return out


def add_group_roll_features(df: pd.DataFrame, key_col: str, prefix: str) -> pd.DataFrame:
    if key_col not in df.columns: return df
    out = df.copy()
    out['__idx__'] = np.arange(len(out))
    grp = out[[key_col, 'pay_time_local', 'online_by_17', '__idx__']].dropna(subset=['pay_time_local'])
    grp = grp.sort_values([key_col, 'pay_time_local'])
    parts = []
    for _, g in grp.groupby(key_col, sort=False):
        g2 = g.set_index('pay_time_local').sort_index()
        mean7 = g2['online_by_17'].shift(1).rolling('7D').mean().rename(f'{prefix}_roll7d_mean')
        cnt7  = g2['online_by_17'].shift(1).rolling('7D').count().rename(f'{prefix}_roll7d_cnt')
        z = pd.concat([mean7, cnt7, g2['__idx__']], axis=1).dropna(subset=['__idx__'])
        parts.append(z)
    if parts:
        zall = pd.concat(parts, axis=0).set_index('__idx__').sort_index()
        out.loc[zall.index, f'{prefix}_roll7d_mean'] = zall[f'{prefix}_roll7d_mean'].astype('float32').values
        out.loc[zall.index, f'{prefix}_roll7d_cnt']  = zall[f'{prefix}_roll7d_cnt' ].astype('float32').values
    return out.drop(columns=['__idx__'])


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy().reset_index(drop=True)
    for col in ['pay_time_local','online_time_local']:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
    df = add_address_clean_and_hist_rate_17(df)
    df = add_window_and_label(df)
    df = add_time_features(df)
    df = add_group_roll_features(df, 'warehouse', 'wh_y17')
    df = add_group_roll_features(df, 'package_type', 'pkg_y17')
    for c in ['customer_name','shipper_name','warehouse','package_type','hour_bin','shipper_address1_clean']:
        if c in df.columns: df[c] = df[c].astype('string').fillna('Missing')
    if 'shipper_state_or_province' in df.columns:
        df['shipper_state_or_province'] = (
            df['shipper_state_or_province']
            .astype('string').str.strip().str.upper().fillna('Missing')
        )
    return df

# ===========================================
# 3) Mixed Encoding（OHE + TargetEncoding）
# ===========================================

def split_categoricals_by_cardinality(df: pd.DataFrame, cat_cols: list[str], threshold: int = 15):
    low, high = [], []
    for c in cat_cols:
        if c not in df.columns: continue
        (low if df[c].nunique(dropna=True) <= threshold else high).append(c)
    return low, high

def _make_ohe():
    kwargs = {"handle_unknown": "ignore", "dtype": np.float32}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = True
    else:
        kwargs["sparse"] = True
    return OneHotEncoder(**kwargs)

def _fit_transform_te_oof(X_train, y_train, X_val, te_cols, te_params):
    import category_encoders as ce
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    Xtr_te = pd.DataFrame(index=X_train.index)
    for tr_idx, oof_idx in kf.split(X_train):
        te = ce.TargetEncoder(cols=te_cols, **te_params)
        _ = te.fit_transform(X_train.iloc[tr_idx][te_cols], y_train.iloc[tr_idx])
        Xtr_te.loc[X_train.iloc[oof_idx].index, te_cols] = te.transform(X_train.iloc[oof_idx][te_cols]).values
    te_full = ce.TargetEncoder(cols=te_cols, **te_params)
    te_full.fit(X_train[te_cols], y_train)
    Xval_te = te_full.transform(X_val[te_cols]) if X_val is not None else None
    return te_full, Xtr_te, Xval_te

def fit_transform_mixed_enc(X_train_raw, y_train, X_val_raw,
                            ohe_cols, te_cols, num_cols,
                            te_params=None):
    te_params = te_params or {"smoothing": 20, "min_samples_leaf": 50,
                              "handle_unknown": "value", "handle_missing": "value"}
    Xtr = X_train_raw.copy()
    Xva = None if X_val_raw is None else X_val_raw.copy()

    if te_cols:
        te, Xtr_te, Xva_te = _fit_transform_te_oof(Xtr, y_train, Xva, te_cols, te_params)
    else:
        te, Xtr_te, Xva_te = None, pd.DataFrame(index=Xtr.index), (pd.DataFrame(index=Xva.index) if Xva is not None else None)

    if ohe_cols:
        ohe = _make_ohe()
        Xtr_ohe = ohe.fit_transform(Xtr[ohe_cols])
        Xva_ohe = ohe.transform(Xva[ohe_cols]) if Xva is not None else None
    else:
        ohe = None
        Xtr_ohe = sparse.csr_matrix((len(Xtr), 0), dtype=np.float32)
        Xva_ohe = (sparse.csr_matrix((len(Xva), 0), dtype=np.float32) if Xva is not None else None)

    medians = {}
    num_cols_present = [c for c in num_cols if c in Xtr.columns]
    Xtr_num = Xtr[num_cols_present].copy().apply(pd.to_numeric, errors='coerce')
    for c in num_cols_present:
        med = Xtr_num[c].median()
        medians[c] = med
        Xtr_num[c] = Xtr_num[c].fillna(med).astype(np.float32)
    if Xva is not None:
        Xva_num = Xva[num_cols_present].copy().apply(pd.to_numeric, errors='coerce')
        for c in num_cols_present:
            Xva_num[c] = Xva_num[c].fillna(medians[c]).astype(np.float32)

    train_blocks = [blk for blk in [Xtr_ohe,
                                    sparse.csr_matrix(Xtr_te.values.astype(np.float32)) if te_cols else None,
                                    sparse.csr_matrix(Xtr_num.values.astype(np.float32)) if num_cols_present else None] if blk is not None]
    X_train_enc = sparse.hstack(train_blocks).tocsr()
    if Xva is not None:
        val_blocks = [blk for blk in [Xva_ohe,
                                      sparse.csr_matrix(Xva_te.values.astype(np.float32)) if te_cols else None,
                                      sparse.csr_matrix(Xva_num.values.astype(np.float32)) if num_cols_present else None] if blk is not None]
        X_val_enc = sparse.hstack(val_blocks).tocsr()
    else:
        X_val_enc = None

    encoders = {"te": te, "ohe": ohe}
    return X_train_enc, X_val_enc, encoders, medians

def transform_mixed_enc(X_raw, encoders, medians, ohe_cols, te_cols, num_cols):
    X = X_raw.copy()
    te  = encoders.get("te");  ohe = encoders.get("ohe")

    X_te  = te.transform(X[te_cols]) if (te_cols and te is not None) else pd.DataFrame(index=X.index)
    X_ohe = ohe.transform(X[ohe_cols]) if (ohe_cols and ohe is not None) else sparse.csr_matrix((len(X), 0), dtype=np.float32)

    num_cols_present = [c for c in num_cols if c in X.columns]
    X_num = X[num_cols_present].copy().apply(pd.to_numeric, errors='coerce')
    for c in num_cols_present:
        X_num[c] = X_num[c].fillna(medians[c]).astype(np.float32)

    blocks = []
    if X_ohe.shape[1] > 0: blocks.append(X_ohe)
    if te_cols:            blocks.append(sparse.csr_matrix(X_te.values.astype(np.float32)))
    if num_cols_present:   blocks.append(sparse.csr_matrix(X_num.values.astype(np.float32)))
    return sparse.hstack(blocks).tocsr()

# ===========================================
# 4) XGBoost + EarlyStopping
# ===========================================

def fit_xgb_earlystop_compat(X_train, y_train, X_val, y_val,
                             scale_pos_weight=1.0,
                             n_estimators=3000, early_rounds=200,
                             params_extra=None):
    params_extra = params_extra or {}
    try:
        from xgboost.callback import EarlyStopping
        model = XGBClassifier(
            n_estimators=n_estimators, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.2, reg_lambda=1.0, min_child_weight=2.0,
            tree_method="hist", objective="binary:logistic",
            eval_metric="logloss", scale_pos_weight=float(scale_pos_weight),
            random_state=42, **params_extra
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            callbacks=[EarlyStopping(rounds=early_rounds, metric_name="logloss",
                                     data_name="validation_1", save_best=True)],
            verbose=True
        )
        def predict_proba_best(X):
            best_it = getattr(model, "best_iteration", None)
            if best_it is not None:
                return model.predict_proba(X, iteration_range=(0, best_it+1))[:, 1]
            return model.predict_proba(X)[:, 1]
        return {"kind":"sklearn_callbacks", "model":model, "predict_proba":predict_proba_best}
    except Exception:
        try:
            model = XGBClassifier(
                n_estimators=n_estimators, learning_rate=0.03, max_depth=6,
                subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.2, reg_lambda=1.0, min_child_weight=2.0,
                tree_method="hist", objective="binary:logistic",
                eval_metric="logloss", scale_pos_weight=float(scale_pos_weight),
                random_state=42, **params_extra
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=early_rounds,
                verbose=True
            )
            def predict_proba_best(X):
                best_it = getattr(model, "best_iteration", None)
                if best_it is not None:
                    return model.predict_proba(X, iteration_range=(0, best_it+1))[:, 1]
                return model.predict_proba(X)[:, 1]
            return {"kind":"sklearn_esr", "model":model, "predict_proba":predict_proba_best}
        except Exception:
            dtrain = xgb.DMatrix(X_train, label=y_train.values.astype(np.float32))
            dvalid = xgb.DMatrix(X_val,   label=y_val.values.astype(np.float32))
            params = {
                "objective":"binary:logistic","eval_metric":"logloss","eta":0.03,"max_depth":6,
                "subsample":0.9,"colsample_bytree":0.9,"reg_alpha":0.2,"reg_lambda":1.0,
                "min_child_weight":2.0,"tree_method":"hist","scale_pos_weight":float(scale_pos_weight),"seed":42
            }
            bst = xgb.train(
                params=params, dtrain=dtrain, num_boost_round=n_estimators,
                evals=[(dtrain,"train"),(dvalid,"valid")],
                early_stopping_rounds=early_rounds, verbose_eval=100
            )
            def predict_proba_best(X):
                dm = xgb.DMatrix(X)
                try: return bst.predict(dm, iteration_range=(0, bst.best_iteration+1))
                except TypeError: return bst.predict(dm, ntree_limit=bst.best_ntree_limit)
            return {"kind":"core_train","model":bst,"predict_proba":predict_proba_best}
        
def fit_xgb_lgb_cat_ensemble_automl(
    X_train, y_train, X_val, y_val,
    sample_weight=None,
    n_trials: int = 30,
    n_estimators: int = 3000,
    early_rounds: int = 200,
    scale_pos_weight: float = 1.0,
):
    """
    XGBoost + LightGBM + CatBoost 三模型 Ensemble
    - XGBoost 走 Optuna 掃參 (準：預設 30 trials)
    - LightGBM 與 CatBoost 用穩健預設（含 early stopping）
    - 回傳介面與既有 bundle 相容：提供 predict_proba（raw，未校準）
    """
    # ---------------------------
    # A) AutoML for XGBoost (Optuna)
    # ---------------------------
    try:
        import optuna
    except ImportError:
        optuna = None

    from xgboost.callback import EarlyStopping as XgbEarlyStop

    best_params = {
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "min_child_weight": 2.0,
    }

    if optuna is not None and len(y_train) > 200:
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 2.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
            }
            mdl = XGBClassifier(
                n_estimators=n_estimators,
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=float(scale_pos_weight),
                random_state=42,
                **params
            )
            mdl.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                callbacks=[XgbEarlyStop(rounds=early_rounds, metric_name="logloss",
                                        data_name="validation_1", save_best=True)],
                verbose=False
            )
            pred = mdl.predict_proba(X_val)[:, 1]
            # 使用 AUC 作為搜尋指標（穩定）
            return roc_auc_score(y_val, pred) if y_val.nunique() > 1 else 0.5

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params.update(study.best_params)

    # 最終 XGBoost（用 best_params）
    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(scale_pos_weight),
        random_state=42,
        **best_params
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[XgbEarlyStop(rounds=early_rounds, metric_name="logloss",
                                data_name="validation_1", save_best=True)],
        verbose=False
    )
    def _xgb_proba(X):
        best_it = getattr(xgb_model, "best_iteration", None)
        if best_it is not None:
            return xgb_model.predict_proba(X, iteration_range=(0, best_it+1))[:, 1]
        return xgb_model.predict_proba(X)[:, 1]

    # ---------------------------
    # B) LightGBM（新版 4.x 用 callbacks 早停）
    # ---------------------------
    lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    lgb_valid = lgb.Dataset(X_val,   label=y_val,   reference=lgb_train)
    lgb_params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.03, num_leaves=63,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        verbose=-1, seed=42
    )
    from lightgbm import early_stopping, log_evaluation
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=n_estimators,
        valid_sets=[lgb_valid],
        callbacks=[early_stopping(stopping_rounds=early_rounds, verbose=False),
                   log_evaluation(0)]
    )
    def _lgb_proba(X):
        return lgb_model.predict(X, num_iteration=lgb_model.best_iteration)

    # ---------------------------
    # C) CatBoost（支援稀疏矩陣 Pool；含早停）
    # ---------------------------
    # CatBoost 直接吃 scipy.sparse（Pool 會處理）
    cat_model = CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
        early_stopping_rounds=early_rounds,
    )
    from catboost import Pool
    cat_train = Pool(X_train, label=y_train, weight=sample_weight)
    cat_valid = Pool(X_val,   label=y_val)
    cat_model.fit(cat_train, eval_set=cat_valid, verbose=False)
    def _cat_proba(X):
        return cat_model.predict_proba(X)[:, 1]

    # ---------------------------
    # D) 三模型加權（0.5 / 0.3 / 0.2）
    # ---------------------------
    def predict_proba_ensemble(X):
        return 0.5*_xgb_proba(X) + 0.3*_lgb_proba(X) + 0.2*_cat_proba(X)

    val_pred_raw = predict_proba_ensemble(X_val)

    return {
        "kind": "xgb+lgb+cat_ensemble_automl",
        "model_xgb": xgb_model,
        "model_lgb": lgb_model,
        "model_cat": cat_model,
        "predict_proba": predict_proba_ensemble,   
        "val_pred_raw": val_pred_raw
    }

# ===========================================
# 5) 視窗與切分
# ===========================================
FORCE_FIXED_WINDOW = True
FORCE_TODAY_DATE = None  

def current_global_window_5am(df: pd.DataFrame, shift_days=0):
    ts = pd.to_datetime(df['pay_time_local'], errors='coerce')
    max_ts = ts.max()
    if pd.isna(max_ts): return pd.NaT, pd.NaT
    end = max_ts.normalize() + pd.Timedelta(hours=5)
    if max_ts < end: end -= pd.Timedelta(days=1)
    end -= pd.Timedelta(days=shift_days)
    start = end - pd.Timedelta(days=1)
    return start, end

def fixed_global_window_5am(today_date: str | None = None):
    if today_date:
        today = pd.to_datetime(today_date).normalize()
    else:
        today = pd.Timestamp.now().normalize()
    end = today + pd.Timedelta(hours=5)
    start = end - pd.Timedelta(days=1)
    return start, end

# ===========================================
# 6) 訓練 / 校準 / 閾值（時間安全）
# ===========================================
CAT_COLS = ['customer_name','shipper_name','warehouse','package_type','hour_bin','shipper_address1_clean']
NUM_COLS = [
    'shipper_address1_historical_rate','box_count','pay_hour','pay_weekday','is_weekend',
    'min_to_cutoff','min_to_deadline_from_pay',
    'wh_y17_roll7d_mean','wh_y17_roll7d_cnt','pkg_y17_roll7d_mean','pkg_y17_roll7d_cnt'
]
TARGET = 'online_by_17'

# ===== TUNABLE CONSTANTS =====
MIN_ROWS_WH = 120      # 分倉門檻的最低驗證樣本數（統一使用這個）
AUC_CUTOFF  = 0.60     # AUC 低於此值 → 不採用分倉門檻
AUC_FULL    = 0.75     # AUC 到達此值 → 分倉權重給滿
VAR_CUTOFF  = 0.02     # 驗證機率標準差過低視為退化

def pick_threshold_f1(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    if len(thr) == 0: return 0.5
    idx = np.argmax(f1[:-1])
    return float(thr[idx])

def train_time_safe(df_feat: pd.DataFrame):
    df = df_feat.copy()
    start, end = fixed_global_window_5am(FORCE_TODAY_DATE) if FORCE_FIXED_WINDOW else current_global_window_5am(df)
    print(f"[INFO] Inference window = [{start}  →  {end})")

    mask_today = (df['pay_time_local'] >= start) & (df['pay_time_local'] < end)
    df_today = df.loc[mask_today].copy()
    df_hist  = df.loc[~mask_today & (df['pay_time_local'] < start)].copy()

    val_start = start - pd.Timedelta(days=3)
    tr = df_hist.loc[df_hist['pay_time_local'] < val_start].dropna(subset=[TARGET]).copy()
    va = df_hist.loc[(df_hist['pay_time_local'] >= val_start) & (df_hist['pay_time_local'] < start)].dropna(subset=[TARGET]).copy()

    X_tr_raw = tr[[c for c in CAT_COLS+NUM_COLS if c in tr.columns]].copy();  y_tr = tr[TARGET].astype(int)
    X_va_raw = va[[c for c in CAT_COLS+NUM_COLS if c in va.columns]].copy();  y_va = va[TARGET].astype(int)

    ohe_cols, te_cols = split_categoricals_by_cardinality(pd.concat([X_tr_raw, X_va_raw], axis=0),
                                                          [c for c in CAT_COLS if c in X_tr_raw.columns], threshold=15)
    print('OHE cols:', ohe_cols)
    print('TE  cols:', te_cols)

    X_tr_enc, X_va_enc, encoders, medians = fit_transform_mixed_enc(
        X_tr_raw, y_tr, X_va_raw,
        ohe_cols=ohe_cols, te_cols=te_cols, num_cols=[c for c in NUM_COLS if c in X_tr_raw.columns],
        te_params={"smoothing": 20, "min_samples_leaf": 50, "handle_unknown": "value", "handle_missing": "value"}
    )

    # === (A) 時間加權（越近越大） ===
    age_days = (start - tr['pay_time_local']).dt.total_seconds() / (3600*24)
    sample_w = np.exp(-0.08 * np.clip(age_days, 0, None)).astype(np.float32)

    # 類別不平衡權重
    pos = int((y_tr==1).sum()); neg = int((y_tr==0).sum())
    scale_pos_weight = max(1.0, neg/max(1,pos))

    # === (B) 三模型 + AutoML（XGB 掃參；LGB/Cat 穩健預設） ===
    bundle = fit_xgb_lgb_cat_ensemble_automl(
        X_tr_enc, y_tr, X_va_enc, y_va,
        sample_weight=sample_w,
        n_trials=30,                   
        n_estimators=3000, early_rounds=200,
        scale_pos_weight=scale_pos_weight
    )
    raw_predict = bundle["predict_proba"]   

    # === (C) Spline Calibration
    val_raw = bundle.get("val_pred_raw", raw_predict(X_va_enc))
    has_both = (y_va.nunique() == 2)
    has_var  = (float(np.nanstd(val_raw)) > 1e-6)

    if has_both and has_var:
        # Step1: 先做 Isotonic 確保單調
        iso1 = IsotonicRegression(out_of_bounds='clip')
        iso1.fit(val_raw, y_va)
        # Step2: 在等距網格上用 Spline 對 isotonic 曲線做平滑
        grid_x = np.linspace(0.0, 1.0, 200)
        grid_y_iso = iso1.transform(grid_x)
        spline = UnivariateSpline(grid_x, grid_y_iso, s=0.001)
        # Step3: 為保單調，再把「樣條平滑後的曲線」送回 Isotonic 做一次投影
        grid_y_smooth = np.clip(spline(grid_x), 0.0, 1.0)
        iso2 = IsotonicRegression(out_of_bounds='clip')
        iso2.fit(grid_x, grid_y_smooth)

        def predict_proba_cal(X, _raw=raw_predict, _spline=spline, _iso2=iso2):
            z = np.clip(_raw(X), 0.0, 1.0)
            return _iso2.transform(np.clip(_spline(z), 0.0, 1.0))

        bundle['calibrator'] = ("spline_iso",)
    else:
        # 小樣本退化：改用 Platt（保守）
        lr = LogisticRegression(max_iter=500)
        lr.fit(val_raw.reshape(-1,1), y_va) if len(val_raw) else None
        def predict_proba_cal(X, _raw=raw_predict, _lr=lr):
            z = _raw(X).reshape(-1,1)
            return _lr.predict_proba(z)[:,1] if hasattr(_lr, "predict_proba") else _raw(X)
        bundle['calibrator'] = ("platt_fallback",)

    bundle['predict_proba_raw'] = raw_predict
    bundle['predict_proba']     = predict_proba_cal

    # === (D) 全域門檻（沿用你的原規則）
    val_proba = predict_proba_cal(X_va_enc)
    if has_both and float(np.nanstd(val_proba)) > 1e-9:
        global_thr = pick_threshold_f1(y_va, val_proba); fallback_used = False
    else:
        base_thr = y_tr.mean() if len(y_tr) > 0 else 0.5
        global_thr = float(np.clip(base_thr, 0.1, 0.9)); fallback_used = True

    print(f"[INFO] Global threshold = {global_thr:.4f} (fallback={'YES' if fallback_used else 'NO'})")

    # === (E) 以下「分倉門檻/收縮/量級對齊/近門檻/Validation/輸出」保持原樣 ===
    va_eval = va.copy()
    va_eval['val_proba'] = val_proba

    thr_by_wh = {}
    for wh, g in va_eval.groupby('warehouse', dropna=False):
        if len(g) >= MIN_ROWS_WH and g[TARGET].nunique() == 2:
            thr_by_wh[wh] = pick_threshold_f1(g[TARGET], g['val_proba'])
    print("[INFO] per-WH thresholds:", {k: round(v,4) for k,v in thr_by_wh.items()})

    auc_by_wh, n_by_wh = {}, {}
    for wh, g in va_eval.groupby('warehouse', dropna=False):
        n_by_wh[wh] = len(g)
        if g[TARGET].nunique() == 2:
            try: auc_by_wh[wh] = roc_auc_score(g[TARGET], g['val_proba'])
            except Exception: auc_by_wh[wh] = np.nan
        else:
            auc_by_wh[wh] = np.nan
    base_by_wh = va_eval.groupby('warehouse')[TARGET].mean().to_dict()
    std_by_wh  = va_eval.groupby('warehouse')['val_proba'].std().to_dict()

    def blended_thr_for_wh(wh):
        thr_wh = thr_by_wh.get(wh, global_thr)
        n   = n_by_wh.get(wh, 0)
        auc = auc_by_wh.get(wh, np.nan)
        base= base_by_wh.get(wh, np.nan)
        s   = std_by_wh.get(wh, 0.0)

        if (n < MIN_ROWS_WH) or (np.isnan(auc) or auc < AUC_CUTOFF) or (s < VAR_CUTOFF) or (base <= 0.03) or (base >= 0.97):
            return float(global_thr)

        w_n  = min(1.0, n / 400.0)
        w_auc = float(np.clip((auc - AUC_CUTOFF) / (AUC_FULL - AUC_CUTOFF), 0.0, 1.0))
        w = w_n * w_auc

        lo = -0.25 * w; hi = +0.12 * w
        thr = global_thr * (1 - w) + thr_wh * w
        return float(np.clip(thr, global_thr + lo, global_thr + hi))

    X_today_raw = df_today[[c for c in CAT_COLS+NUM_COLS if c in df_today.columns]].copy()
    X_today_enc = transform_mixed_enc(
        X_today_raw, encoders, medians,
        ohe_cols=ohe_cols, te_cols=te_cols, num_cols=[c for c in NUM_COLS if c in X_today_raw.columns]
    )
    proba_today = predict_proba_cal(X_today_enc)

    RECENT_DAYS = 7
    K_SHRINK    = 200
    MIN_TODAY_N = 30
    RATE_CLIP   = (0.03, 0.97)

    recent_mask = df_hist['pay_time_local'] >= (start - pd.Timedelta(days=RECENT_DAYS))
    recent = df_hist.loc[recent_mask].dropna(subset=[TARGET]).copy()
    global_rate_recent = float(recent[TARGET].mean()) if len(recent) else float(y_tr.mean())
    wh_stats = (recent.groupby('warehouse', dropna=False)[TARGET]
                    .agg(['mean','count']).rename(columns={'mean':'rate','count':'n'}))
    if wh_stats.empty:
        wh_stats = pd.DataFrame(columns=['rate','n'])

    def shrunk_rate(wh):
        if wh in wh_stats.index:
            r = float(wh_stats.loc[wh,'rate']); n = float(wh_stats.loc[wh,'n'])
            return float((r*n + global_rate_recent*K_SHRINK) / (n + K_SHRINK))
        return float(global_rate_recent)

    df_today_out = df_today.copy()
    df_today_out['pred_proba_online_by_17'] = proba_today
    df_today_out['applied_threshold'] = df_today_out['warehouse'].apply(blended_thr_for_wh)

    def tune_threshold_for_group(g):
        wh = g.name
        n_today = len(g)
        if n_today < MIN_TODAY_N:
            return g['applied_threshold']
        target_rate = np.clip(shrunk_rate(wh), *RATE_CLIP)
        q_thr = float(np.quantile(g['pred_proba_online_by_17'].values, 1.0 - target_rate))
        w = min(1.0, (wh_stats.loc[wh,'n'] if wh in wh_stats.index else 0) / 400.0)
        thr0 = float(np.median(g['applied_threshold']))
        thr_final = thr0*(1-w) + q_thr*w
        thr_final = float(np.clip(thr_final, thr0 - 0.10, thr0 + 0.10))
        return pd.Series(np.full(n_today, thr_final), index=g.index, dtype='float64')

    df_today_out['applied_threshold'] = (
        df_today_out.groupby('warehouse', group_keys=False).apply(tune_threshold_for_group)
    )
    df_today_out['pred_online_by_17'] = (df_today_out['pred_proba_online_by_17'] >= df_today_out['applied_threshold']).astype(int)

    df_today_out['near_thr'] = (
        (df_today_out['pred_proba_online_by_17'] >= df_today_out['applied_threshold'] - 0.02) &
        (df_today_out['pred_proba_online_by_17'] <= df_today_out['applied_threshold'] + 0.02)
    ).astype(int)

    va_eval = va.copy(); va_eval['val_proba'] = val_proba
    va_eval['pred'] = (va_eval['val_proba'] >= global_thr).astype(int)
    rows = len(va_eval)
    overall = [{
        'scope': 'overall',
        'rows': rows,
        'acc': accuracy_score(va_eval[TARGET], va_eval['pred']),
        'f1':  f1_score(va_eval[TARGET], va_eval['pred']),
        'auc': roc_auc_score(va_eval[TARGET], va_eval['val_proba']) if va_eval[TARGET].nunique()>1 else np.nan,
        'thr': global_thr
    }]
    by_wh = []
    for wh, g in va_eval.groupby('warehouse'):
        by_wh.append({
            'scope': f'wh:{wh}',
            'rows': len(g),
            'acc': accuracy_score(g[TARGET], g['pred']),
            'f1':  f1_score(g[TARGET], g['pred']) if g[TARGET].nunique()>1 else np.nan,
            'auc': roc_auc_score(g[TARGET], g['val_proba']) if g[TARGET].nunique()>1 else np.nan,
            'thr': global_thr
        })
    val_df = pd.DataFrame(overall + by_wh)

    print("\n[DIAG] Today per-WH distribution & thresholds")
    for wh, g in df_today_out.groupby('warehouse'):
        qs = np.quantile(g['pred_proba_online_by_17'], [0.1, 0.5, 0.9])
        print(f"{str(wh):12s} n={len(g):4d} q10={qs[0]:.3f} q50={qs[1]:.3f} q90={qs[2]:.3f} "
              f"thr≈{np.median(g['applied_threshold']):.3f}  pred_rate={g['pred_online_by_17'].mean():.3f}")

    artifacts = {
        'bundle': bundle,
        'encoders': encoders,
        'medians': medians,
        'ohe_cols': ohe_cols,
        'te_cols': te_cols,
        'global_thr': global_thr,
        'thr_by_wh': thr_by_wh,
        'df_today_out': df_today_out,
        'val_df': val_df,
        'df_today': df_today,
        'df_hist': df_hist,
        'va_eval': va_eval
    }
    return artifacts

# ===========================================
# 7) 今天視窗彙總 + 圖表
# ===========================================
def make_today_summary(df_today_out: pd.DataFrame) -> pd.DataFrame:
    if df_today_out.empty:
        print("今天視窗沒有資料")
        return df_today_out
    df_today_out = df_today_out.copy()
    if 'shipper_state_or_province' in df_today_out.columns:
        df_today_out['shipper_state_or_province'] = (
            df_today_out['shipper_state_or_province']
            .astype('string').str.strip().str.upper().fillna('Missing')
        )
    detail = (
        df_today_out
        .groupby(['shipper_state_or_province','shipper_address1_clean'], dropna=False)
        .agg(
            Predicted_Pickup_Rate=('pred_online_by_17','mean'),
            Mean_Probability=('pred_proba_online_by_17','mean'),
            Actual_order_volume=('pred_online_by_17','size'),
            Forecast_pickup_volume=('pred_online_by_17','sum')
        ).reset_index()
    ).sort_values(['shipper_state_or_province','shipper_address1_clean']).reset_index(drop=True)
    subs = []
    for st, g in detail.groupby('shipper_state_or_province', dropna=False):
        cnt  = int(g['Actual_order_volume'].sum())
        pred = int(g['Forecast_pickup_volume'].sum())
        rate = (pred/cnt) if cnt>0 else np.nan
        subs.append({
            'shipper_state_or_province': st,
            'shipper_address1_clean': 'Subtotal',
            'Predicted_Pickup_Rate': rate,
            'Mean_Probability': g['Mean_Probability'].mean(),
            'Actual_order_volume': cnt,
            'Forecast_pickup_volume': pred
        })
    subtotal = pd.DataFrame(subs)
    g_cnt = int(detail['Actual_order_volume'].sum())
    g_pred= int(detail['Forecast_pickup_volume'].sum())
    g_rate= (g_pred/g_cnt) if g_cnt>0 else np.nan
    total = pd.DataFrame([{
        'shipper_state_or_province':'',
        'shipper_address1_clean':'Grand Total',
        'Predicted_Pickup_Rate': g_rate,
        'Mean_Probability': detail['Mean_Probability'].mean(),
        'Actual_order_volume': g_cnt,
        'Forecast_pickup_volume': g_pred
    }])
    out = []
    for st, g in detail.groupby('shipper_state_or_province', dropna=False):
        out.append(g); out.append(subtotal[subtotal['shipper_state_or_province'].eq(st)])
    final_out = pd.concat(out + [total], ignore_index=True)
    final_out['Predicted_Pickup_Rate'] = final_out['Predicted_Pickup_Rate'].round(6)
    return final_out

def plot_summary_by_address(summary_df: pd.DataFrame,
                            out_path="outputs/summary_by_address2.png",
                            top_n=None,
                            group_by_state=True,
                            within_state_sort="rate_desc"):
    if summary_df is None or summary_df.empty: return None
    dfp = summary_df[~summary_df["shipper_address1_clean"].isin(["Subtotal", "Grand Total"])].copy()
    for c in ["Predicted_Pickup_Rate","Actual_order_volume","Forecast_pickup_volume"]:
        if c in dfp.columns: dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0)
    dfp["shipper_state_or_province"] = (
    dfp["shipper_state_or_province"]
    .fillna("").astype(str).str.strip().str.upper()  # ← 加上 .str.upper()
)

    dfp["shipper_address1_clean"]    = dfp["shipper_address1_clean"].fillna("").astype(str).str.strip()
    if group_by_state:
        if within_state_sort == "alpha":
            dfp = dfp.sort_values(["shipper_state_or_province","shipper_address1_clean"], ascending=[True,True])
        else:
            dfp = dfp.sort_values(["shipper_state_or_province","Predicted_Pickup_Rate","Actual_order_volume"],
                                  ascending=[True,False,False])
    else:
        dfp = dfp.sort_values(["Predicted_Pickup_Rate","Actual_order_volume"], ascending=[False,False])
    if isinstance(top_n, int) and top_n>0:
        dfp = dfp.head(top_n)
    dfp = dfp.iloc[::-1]
    ylabels = (dfp["shipper_state_or_province"] + " | " + dfp["shipper_address1_clean"]).tolist()
    actual  = dfp["Actual_order_volume"].astype(int).to_numpy()
    fcst    = dfp["Forecast_pickup_volume"].astype(int).to_numpy()
    vals    = np.where(actual>0, fcst/actual, 0.0)
    h_inches = max(6, 0.35*len(dfp))
    fig, ax = plt.subplots(figsize=(11, h_inches))
    y = np.arange(len(vals))
    ax.barh(y, vals)
    ax.set_xlim(0,1)
    ax.set_yticks(y); ax.set_yticklabels(ylabels)
    ax.set_xlabel("Predicted Pickup Rate")
    ax.set_title("Predicted Pickup Rate by Address   (right: Actual/Forecast)")
    for i, (v,a,f) in enumerate(zip(vals, actual, fcst)):
        ax.text(min(1.0, v+0.01), i, f"{int(round(v*100))}% | {a}/{f}", va="center", ha="left", fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180); plt.close(fig)
    return out_path

def plot_summary_by_state(summary_df: pd.DataFrame, out_path="outputs/summary_by_state.png"):
    if summary_df is None or summary_df.empty: return None
    summary_df = summary_df.copy()
    summary_df["shipper_state_or_province"] = (
        summary_df["shipper_state_or_province"]
        .fillna("").astype(str).str.strip().str.upper()
    )
    sub = summary_df[summary_df["shipper_address1_clean"].eq("Subtotal")].copy()
    if sub.empty:
        g = summary_df[~summary_df["shipper_address1_clean"].isin(["Subtotal","Grand Total"])].copy()
        g["Predicted_Pickup_Rate"] = pd.to_numeric(g["Predicted_Pickup_Rate"], errors="coerce")
        g["Actual_order_volume"]   = pd.to_numeric(g["Actual_order_volume"], errors="coerce")
        g["Forecast_pickup_volume"]= pd.to_numeric(g["Forecast_pickup_volume"], errors="coerce")
        sub = (g.groupby("shipper_state_or_province", dropna=False)
                 .agg(Predicted_Pickup_Rate=("Predicted_Pickup_Rate","mean"),
                      Actual_order_volume=("Actual_order_volume","sum"),
                      Forecast_pickup_volume=("Forecast_pickup_volume","sum"))
                 .reset_index())
        sub["shipper_address1_clean"] = "Subtotal"
    sub = sub.sort_values(["Predicted_Pickup_Rate","Actual_order_volume"], ascending=[False,False]).iloc[::-1]
    ylabels = sub["shipper_state_or_province"].astype(str).tolist()
    actual  = pd.to_numeric(sub["Actual_order_volume"], errors="coerce").fillna(0).astype(int).to_numpy()
    fcst    = pd.to_numeric(sub["Forecast_pickup_volume"], errors="coerce").fillna(0).astype(int).to_numpy()
    vals    = np.where(actual>0, fcst/actual, 0.0)
    fig, ax = plt.subplots(figsize=(8,6))
    y = np.arange(len(vals))
    ax.barh(y, vals)
    ax.set_xlim(0,1)
    ax.set_yticks(y); ax.set_yticklabels(ylabels)
    ax.set_xlabel("Predicted Pickup Rate")
    ax.set_title("Predicted Pickup Rate by State   (right: Actual/Forecast)")
    for i, (v,a,f) in enumerate(zip(vals, actual, fcst)):
        ax.text(min(1.0, v+0.01), i, f"{int(round(v*100))}% | {a}/{f}", va="center", ha="left", fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180); plt.close(fig)
    return out_path

def _sanitize_filename(s: str) -> str:
    s = (s or "NA").strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def plot_summary_by_address_per_state(summary_df: pd.DataFrame,
                                      out_dir="outputs/charts_by_state2",
                                      within_state_sort="rate_desc") -> dict:
    paths = {}
    if summary_df is None or summary_df.empty: return paths
    dfp = summary_df[~summary_df["shipper_address1_clean"].isin(["Subtotal","Grand Total"])].copy()
    for c in ["Predicted_Pickup_Rate","Actual_order_volume","Forecast_pickup_volume"]:
        if c in dfp.columns: dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0)
    dfp["shipper_state_or_province"] = (
    dfp["shipper_state_or_province"]
    .fillna("").astype(str).str.strip().str.upper()  # ← 加上 .str.upper()
)
    dfp["shipper_address1_clean"]    = dfp["shipper_address1_clean"].fillna("").astype(str).str.strip()
    states = sorted(dfp["shipper_state_or_province"].unique().tolist())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for st in states:
        g = dfp.loc[dfp["shipper_state_or_province"].eq(st)].copy()
        if g.empty: continue
        if within_state_sort == "alpha":
            g = g.sort_values(["shipper_address1_clean"], ascending=[True])
        else:
            g = g.sort_values(["Predicted_Pickup_Rate","Actual_order_volume"], ascending=[False,False])
        g = g.iloc[::-1]
        ylabels = g["shipper_address1_clean"].tolist()
        vals    = g["Predicted_Pickup_Rate"].clip(0,1).to_numpy()
        actual  = g["Actual_order_volume"].astype(int).to_numpy()
        fcst    = g["Forecast_pickup_volume"].astype(int).to_numpy()
        h_inches = max(4, 0.35*len(g))
        fig, ax = plt.subplots(figsize=(10, h_inches))
        y = np.arange(len(vals))
        ax.barh(y, vals)
        ax.set_xlim(0,1)
        ax.set_yticks(y); ax.set_yticklabels(ylabels)
        ax.set_xlabel("Predicted Pickup Rate")
        title_state = st if st else "(blank)"
        ax.set_title(f"{title_state} – Predicted Pickup Rate by Address   (right: Actual/Forecast)")
        for i, (v,a,f) in enumerate(zip(vals, actual, fcst)):
            ax.text(min(1.0, v+0.01), i, f"{int(round(v*100))}% | {a}/{f}", va="center", ha="left", fontsize=9)
        plt.tight_layout()
        fpath = str(Path(out_dir) / f"{_sanitize_filename(title_state)}.png")
        fig.savefig(fpath, dpi=180); plt.close(fig)
        paths[st] = fpath
    return paths

# ===========================================
# 8) 7 日回測（只用來平滑 global_thr）
# ===========================================
def run_backtest_7d(df_feat: pd.DataFrame):
    df = df_feat.copy()
    if FORCE_FIXED_WINDOW:
        _, base_end = fixed_global_window_5am(FORCE_TODAY_DATE)
    else:
        _, base_end = current_global_window_5am(df)
    results = []
    for back in range(7, 0, -1):  # D-7 ... D-1
        bt_end = base_end - pd.Timedelta(days=back)
        bt_start = bt_end - pd.Timedelta(days=1)
        val_mask = (df['pay_time_local'] >= bt_start) & (df['pay_time_local'] < bt_end)
        train_mask = (df['pay_time_local'] < bt_start)
        tr = df.loc[train_mask].dropna(subset=[TARGET]).copy()
        va = df.loc[val_mask].dropna(subset=[TARGET]).copy()
        if len(va) < 50 or len(tr) < 200: continue
        X_tr_raw = tr[[c for c in CAT_COLS+NUM_COLS if c in tr.columns]].copy();  y_tr = tr[TARGET].astype(int)
        X_va_raw = va[[c for c in CAT_COLS+NUM_COLS if c in va.columns]].copy();  y_va = va[TARGET].astype(int)
        ohe_cols, te_cols = split_categoricals_by_cardinality(pd.concat([X_tr_raw, X_va_raw], axis=0),
                                                              [c for c in CAT_COLS if c in X_tr_raw.columns], threshold=15)
        X_tr_enc, X_va_enc, encoders, medians = fit_transform_mixed_enc(
            X_tr_raw, y_tr, X_va_raw,
            ohe_cols=ohe_cols, te_cols=te_cols, num_cols=[c for c in NUM_COLS if c in X_tr_raw.columns]
        )
        pos = int((y_tr==1).sum()); neg = int((y_tr==0).sum())
        spw = max(1.0, neg/max(1,pos))
        bundle = fit_xgb_earlystop_compat(X_tr_enc, y_tr, X_va_enc, y_va, scale_pos_weight=spw,
                                          n_estimators=2500, early_rounds=150)
        raw_predict = bundle['predict_proba']
        raw = raw_predict(X_va_enc)
        has_both = (y_va.nunique() == 2)
        has_var  = (float(np.nanstd(raw)) > 1e-6)
        if has_both and has_var:
            iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(raw, y_va)
            proba = iso.transform(raw); thr = pick_threshold_f1(y_va, proba)
        else:
            proba = raw; base_thr = y_tr.mean() if len(y_tr) > 0 else 0.5
            thr = float(np.clip(base_thr, 0.1, 0.9))
        pred = (proba >= thr).astype(int)
        results.append({
            'day_back': back,
            'rows': len(va),
            'acc': accuracy_score(y_va, pred),
            'f1':  f1_score(y_va, pred) if y_va.nunique()>1 else np.nan,
            'auc': roc_auc_score(y_va, proba) if y_va.nunique()>1 else np.nan,
            'thr': thr
        })
    return pd.DataFrame(results).sort_values(['day_back'])

# ===========================================
# 9) 主流程
# ===========================================
def main():
    out_dir = Path("./outputs"); out_dir.mkdir(exist_ok=True)

    # A) MySQL → DataFrame
    print("[STEP] Reading MySQL ...")
    QUERY = QUERY_TMPL_ALL
    df_raw = read_mysql_to_df(QUERY, chunksize=None)
    print(f"[INFO] raw rows = {len(df_raw)}")

    # B) 特徵工程
    print("[STEP] Building features ...")
    df_feat = build_features(df_raw)
    df_feat.to_parquet(out_dir/"mysql_featured.parquet", index=False)
    print(f"[INFO] features rows={len(df_feat)}, cols={len(df_feat.columns)}")

    # C) 時間安全訓練與今天推論
    artifacts = train_time_safe(df_feat)

    # 先取出今天結果
    df_today_out = artifacts['df_today_out'].copy()

    # D) 7日回測：只平滑 global_thr
    bt = run_backtest_7d(df_feat)
    if not bt.empty:
        bt_rep = np.repeat(bt['thr'].to_numpy(), bt['rows'].astype(int).to_numpy())
        thr_smoothed = float(np.clip(np.median(bt_rep), 0.3, 0.8))
        artifacts['global_thr'] = thr_smoothed
        artifacts['df_today_out'] = df_today_out  

    # E) 彙總 + 匯出
    summary_df = make_today_summary(artifacts['df_today_out'])
    out_excel = out_dir / "Today_Forecast_Address_Summary1217.xlsx"
    with pd.ExcelWriter(out_excel, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        artifacts['df_today_out'].to_excel(writer, index=False, sheet_name="Detail")
        artifacts['val_df'].to_excel(writer, index=False, sheet_name="Validation")

        va_eval = artifacts['va_eval']
        thr_rows = []
        for wh, g in va_eval.groupby('warehouse', dropna=False):
            auc = roc_auc_score(g[TARGET], g['val_proba']) if g[TARGET].nunique()==2 else np.nan
            thr_rows.append({
                'warehouse': wh,
                'rows_va': len(g),
                'base_rate_va': float(g[TARGET].mean()) if len(g)>0 else np.nan,
                'auc_va': float(auc) if not np.isnan(auc) else np.nan,
                'thr_wh_raw': float(artifacts['thr_by_wh'].get(wh, np.nan)),
                'thr_applied_today_median': float(
                    artifacts['df_today_out'].loc[artifacts['df_today_out']['warehouse'].eq(wh), 'applied_threshold'].median()
                ) if not artifacts['df_today_out'].empty else np.nan
            })
        pd.DataFrame(thr_rows).sort_values('rows_va', ascending=False)\
            .to_excel(writer, index=False, sheet_name="Thresholds_By_WH")
        pd.DataFrame([{'global_thr': artifacts['global_thr']}])\
            .to_excel(writer, index=False, sheet_name="Thresholds_Global")

        # 圖
        img_all = plot_summary_by_address(summary_df, out_path=str(out_dir / "summary_by_address2.png"),
                                          top_n=None, group_by_state=True, within_state_sort="rate_desc")
        img_state = plot_summary_by_state(summary_df, out_path=str(out_dir / "summary_by_state2.png"))
        per_state_imgs = plot_summary_by_address_per_state(summary_df,
                                                           out_dir=str(out_dir / "charts_by_state2"),
                                                           within_state_sort="rate_desc")

        ws_summary = writer.sheets["Summary"]
        if img_all:   ws_summary.insert_image("H2",  str(img_all),   {"x_scale": 0.9, "y_scale": 0.9})
        if img_state: ws_summary.insert_image("H35", str(img_state), {"x_scale": 0.9, "y_scale": 0.9})

        wb = writer.book
        ws_by_state = wb.add_worksheet("Charts_By_State2")
        row = 1; scale = 0.9
        for st in sorted(per_state_imgs.keys(), key=lambda x: (str(x) if x is not None else "")):
            ws_by_state.write(row, 0, f"State: {st if st else '(blank)'}")
            ws_by_state.insert_image(row + 1, 0, per_state_imgs[st], {"x_scale": scale, "y_scale": scale})
            row += 38
        print(f"\n已輸出：{out_excel}")

    out_txt = out_dir / "Today_Forecast_Address_Summary1217.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== Summary ===\n")
        f.write(summary_df.to_string(index=False) if not summary_df.empty else "(no rows)")
    print(f"已輸出 TXT：{out_txt}")

if __name__ == "__main__":
    main()
