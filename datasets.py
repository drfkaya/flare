"""
KALE Benchmark Datasets — Düzeltildi
=====================================
Heart ve Ionosphere OpenML düzeltildi.
Linear sentetik veri eklendi.
Ground truth coefficients meta'ya eklendi (SIM, LINEAR).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, load_iris,
    fetch_openml
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def load_dataset(name, normalize=True, seed=42):
    """Veri seti yükle + ön işleme. Returns: X, y, meta"""
    name = name.lower().strip()
    """
    loaders = {
        'breast_cancer': _load_breast_cancer,
        'wine':          _load_wine,
        'digits':        _load_digits,
        'iris':          _load_iris,
        'heart':         _load_heart,
        'diabetes':      _load_diabetes,
        'ionosphere':    _load_ionosphere,
        'australian':    _load_australian,
        'sim':           _load_sim,
        'linear':        _load_linear,
        'support2':     _load_support2
    }
    """
    # load_dataset içindeki loaders dict'ine ekle:
    loaders = {
        'breast_cancer': _load_breast_cancer,
        'diabetes':      _load_diabetes,
        'australian':    _load_australian,
        'sim':           _load_sim,
        'linear':        _load_linear,
        'support2':     _load_support2,
        'metabric':    _load_metabric,
        'flchain': _load_flchain,
        'vtc':_load_vtc,
        'fic':_load_fic,
        'dor_model1': lambda seed=42: _load_dor(seed, comparison=2, li_features=True),
        'dor_model2': lambda seed=42: _load_dor(seed, comparison=3, li_features=True),
        'dor_comp2':  lambda seed=42: _load_dor(seed, comparison=2, li_features=False),
        'dor_comp3':  lambda seed=42: _load_dor(seed, comparison=3, li_features=False),
        # ── KRAL (sürekli) ──
        # ── CLNM target ──
        'ptc_clnm':      lambda seed=42: _load_ptc(seed, 'CLNM', False),
        'ptc_clnm_li':   lambda seed=42: _load_ptc(seed, 'CLNM', True),

        # ── LLNM target + CLNM predictor (Li Model 2 ile aynı) ──
        'ptc_llnm_clnm':      lambda seed=42: _load_ptc(seed, 'LLNM_CLNM', False),
        'ptc_llnm_clnm_li':   lambda seed=42: _load_ptc(seed, 'LLNM_CLNM', True),
        'eicu': _load_eicu




        


    }


    if name not in loaders:
        raise ValueError(
            f"'{name}' bulunamadı. "
            f"Geçerli: {', '.join(loaders.keys())}")

    X, y, meta = loaders[name](seed=seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    # Binary kontrol
    if len(np.unique(y)) != 2:
        raise ValueError(f"Binary etiket gerekli, "
                         f"{len(np.unique(y))} sınıf bulundu")

    # NaN/Inf temizliği
    mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
    if mask.any():
        meta['n_dropped_nan'] = int(mask.sum())
        X, y = X[~mask], y[~mask]
    else:
        meta['n_dropped_nan'] = 0

    # Normalizasyon
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        meta['normalized'] = True
    else:
        meta['normalized'] = False

    meta['n_final'] = len(y)
    meta['d_final'] = X.shape[1]
    meta['pos_rate'] = float(y.mean())
    meta['class_0'] = int((y == 0).sum())
    meta['class_1'] = int((y == 1).sum())

    return X, y, meta


# =====================================================================
# SKLEARN DATASETS
# =====================================================================
def _load_eicu(seed=42, subgroup='sepsis', feature_set='focused'):
    """
    GOSSIS-1-eICU — İlk 24 saat ICU verileri.

    Parameters
    ----------
    subgroup : str
        'all', 'sepsis', 'vent', 'sepsis_vent', 'floor', 'interm'
    feature_set : str
        'full'      → tüm öznitelikler (d≈150-600)
        'focused'   → β'e dayalı klinik öznitelikler (d≈25)
        'optimized' → β + temizlenmiş kategorikler (d≈40-50)
    """
    import os
    csv_path = os.path.join(os.getcwd(), 'eicu.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"eicu.csv bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path, na_values=['?', ''])

    # ── ALT GRUP FİLTRE ──
    filters = {
        'all': pd.Series(True, index=df.index),
        'sepsis': df['group'] == 'Sepsis',
        'vent': df['vent'] == 1,
        'sepsis_vent': (df['group'] == 'Sepsis') & (df['vent'] == 1),
        'floor': df['icu_admit_source'] == 'Floor',
        'interm': df['dcs_group'] == 'INTERM_DCS',
    }
    df = df[filters[subgroup]].reset_index(drop=True)

    # ── HEDEF + PARTITION ──
    y = df['hospital_death'].astype(np.float64).values
    train_mask = (df['partition'] == 'training').values
    test_mask = (df['partition'] == 'testing').values

    # ═══════════════════════════════════════════════════════
    # FEATURE SET: optimized
    # ═══════════════════════════════════════════════════════
    if feature_set == 'optimized':

        # ── 1. Sayısal fizyolojik (β'e göre sıralı) ──
        physio_avg = [
            'd1_lactate_avg',       # β=0.598 ***
            'd1_albumin_avg',       # β=-0.360 ***
            'd1_hemaglobin_avg',    # β=-0.323 ***
            'd1_bun_avg',           # β=0.298 ***
            'd1_spo2_avg',          # β=-0.301 ***
            'd1_hematocrit_avg',    # β=0.262 ***
            'd1_sysbp_avg',         # β=-0.205 ***
            'd1_mbp_avg',           # β=-0.181 ***
            'd1_calcium_avg',       # β=0.169 ***
            'd1_resprate_avg',      # β=0.168 ***
            'd1_temp_avg',          # β=-0.164 ***
            'd1_bilirubin_avg',     # β=0.133 ***
            'd1_heartrate_avg',     # β=0.135 ***
            'd1_potassium_avg',     # β=0.088 **
            'd1_sodium_avg',        # β=-0.081 **
            'd1_platelets_avg',     # β=-0.080 *
            'd1_diasbp_avg',        # β=0.092 .
            'd1_creatinine_avg',    # β=-0.081 .
        ]

        physio_diff = [
            'd1_heartrate_diff',    # β=0.077 *
            'd1_inr_diff',          # β=0.072 *
            'd1_bun_diff',          # β=-0.080 *
            'd1_calcium_diff',      # β=-0.073 *
        ]

        # ── 2. Klinik binary ──
        clinical = [
            'age',                  # β=0.407 ***
            'vent',                 # β=0.343 ***
            'solid_tumor_with_metastasis',  # β=0.123 ***
            'diabetes_mellitus',    # β=-0.123 ***
            'arf_apache',           # β=0.083 *
            'cirrhosis',            # β=0.059 *
        ]

        # ── 3. Kategorik — SADECE ANLAMLI LEVEL'LAR ──
        # dx_class: 502Sepsis (β=-0.160, p<0.001)
        if 'dx_class' in df.columns:
            df['dx_is_502'] = (
                df['dx_class'] == '502Sepsis').astype(float)
            # 501Sepsis at (p=0.229)

        # dcs_group: ALL_LOW, INTERM, SOME_HIGH tut
        #            ALL_HIGH at (p=0.110) — reference
        if 'dcs_group' in df.columns:
            df['dcs_all_low'] = (
                df['dcs_group'] == 'ALL_LOW').astype(float)
            df['dcs_interm'] = (
                df['dcs_group'] == 'INTERM_DCS').astype(float)
            df['dcs_some_high'] = (
                df['dcs_group'] == 'SOME_HIGH').astype(float)
            # ALL_HIGH = reference (0,0,0)

        # icu_admit_source: Floor + Other Hospital
        #                    A&E, OR, Other ICU at
        if 'icu_admit_source' in df.columns:
            df['src_floor'] = (
                df['icu_admit_source'] == 'Floor').astype(float)
            df['src_other_hosp'] = (
                df['icu_admit_source'] == 'Other Hospital').astype(float)
            # A&E, OR, Other ICU = reference (0,0)

        # group AT (hepsi Sepsis → sabit)
        # dx_sub AT (7 level, 6'sı anlamsız)

        # ── 4. Birleştir ──
        optimized_cats = [
            'dx_is_502',
            'dcs_all_low', 'dcs_interm', 'dcs_some_high',
            'src_floor', 'src_other_hosp',
        ]

        feature_cols = physio_avg + physio_diff + clinical + optimized_cats
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols].astype(np.float64).values

    # ═══════════════════════════════════════════════════════
    # FEATURE SET: focused (sadece sayısal)
    # ═══════════════════════════════════════════════════════
    elif feature_set == 'focused':
        physio_avg = [
            'd1_lactate_avg', 'd1_albumin_avg',
            'd1_hemaglobin_avg', 'd1_bun_avg',
            'd1_spo2_avg', 'd1_hematocrit_avg',
            'd1_sysbp_avg', 'd1_mbp_avg',
            'd1_calcium_avg', 'd1_resprate_avg',
            'd1_temp_avg', 'd1_bilirubin_avg',
            'd1_heartrate_avg', 'd1_potassium_avg',
            'd1_sodium_avg',
        ]
        physio_diff = [
            'd1_heartrate_diff', 'd1_inr_diff',
            'd1_bun_diff', 'd1_calcium_diff',
        ]
        clinical = [
            'age', 'vent', 'solid_tumor_with_metastasis',
            'diabetes_mellitus', 'arf_apache', 'cirrhosis',
        ]
        feature_cols = physio_avg + physio_diff + clinical
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols].astype(np.float64).values

    # ═══════════════════════════════════════════════════════
    # FEATURE SET: full (orijinal)
    # ═══════════════════════════════════════════════════════
    else:
        drop_cols = [
            'patientunitstayid', 'encounter_id', 'partition',
            'hospital_death', 'icu_death',
        ]
        cat_cols = [
            'dx_class', 'dx_sub', 'dcs_group',
            'icu_admit_source', 'group',
        ]
        for col in cat_cols:
            if col in df.columns:
                dummies = pd.get_dummies(
                    df[col], prefix=col, dummy_na=False)
                df = pd.concat([df, dummies], axis=1)
                drop_cols.append(col)
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].astype(np.float64).values

    # ── İMPUTE ──
    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            col[miss] = np.nanmedian(col[~miss])

    # ── TEMİZLİK ──
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    train_mask = train_mask[mask]
    test_mask = test_mask[mask]

    return X, y, {
        'name': f'eICU [{subgroup}/{feature_set}]',
        'source': 'PhysioNet GOSSIS-1-eICU (Raffa et al., 2022)',
        'n_final': len(y), 'd_final': X.shape[1],
        'pos_rate': y.mean(),
        'feature_names': feature_cols,
        'real_data': True,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'subgroup': subgroup,
        'feature_set': feature_set,
    }

def _load_ptc(seed=42, target='CLNM', li_features=False):
    """
    PTC — Papillary Thyroid Cancer, LNM Prediction
    Kaynak: Single center, n≈2428
    Dosya: ptc.xlsx (aynı dizinde, Excel)

    target='CLNM'  → Central lymph node metastasis
    target='LLNM'  → Lateral lymph node metastasis
    target='LLNM_CLNM' → LLNM, CLNM predictor olarak dahil

    li_features=True  → Li et al.'ın cutoff'ları
    li_features=False → Sürekli + tüm feature'lar
    """
    xlsx_path = os.path.join(os.getcwd(), 'ptc.xlsx')
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"ptc.xlsx bulunamadı: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    first_sheet = xls.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=first_sheet)
    print(f"  Sheet: '{first_sheet}', {len(df)} satır")

    if 'number' in df.columns:
        df = df.drop_duplicates(subset=['number'], keep='first')
    print(f"  n = {len(df)}")

    df.columns = df.columns.str.strip()

    # ── VİRGÜL ONDALIK ──
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (df[col].astype(str)
                       .str.replace(',', '.', regex=False)
                       .str.replace('，', '.', regex=False))
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > 0.3 * len(df):
                df[col] = converted

    # ── TARGET ──
    # LLNM_CLNM → target LLNM ama CLNM feature olarak kalır
    actual_target = target
    include_clnm = False
    if target.upper() == 'LLNM_CLNM':
        actual_target = 'LLNM'
        include_clnm = True

    target_map = {'CLNM': 'CLNM', 'LLNM': 'LLMN'}
    target_col = target_map.get(actual_target.upper(), actual_target.upper())
    if target_col not in df.columns:
        for col in df.columns:
            col_upper = col.upper().strip()
            if col_upper in ('LLMN', 'LLNM'):
                if 'LLNM' in actual_target.upper():
                    target_col = col
                    break
            if col_upper == 'CLNM' and actual_target.upper() == 'CLNM':
                target_col = col
                break
    print(f"  Target: '{target_col}' "
          f"(1={(df[target_col]==1).sum()}, "
          f"0={(df[target_col]==0).sum()})")
    y = df[target_col].astype(np.float64).values

    # ── SIZINTI AT (ama CLNM'yi koru eğer gerekliyse) ──
    drop_cols = ['number', 'LLMN', 'LLNM']
    if not include_clnm:
        drop_cols.append('CLNM')    # target değilse CLNM'yi de at
    else:
        # CLNM feature olarak kalacak, sadece rename et
        pass
    df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                 errors='ignore')

    # ── GENDER ENCODING ──
    if 'gender' in df.columns:
        df['male'] = (df['gender'].astype(str).str.strip().str.lower()
                      .map({'male': 1, 'female': 0}))
        df = df.drop(columns=['gender'])

    # ── LOCATION TEMİZLEME ──
    if 'location' in df.columns:
        df['location'] = (df['location'].astype(str)
                          .str.strip().str.lower()
                          .str.replace('、', ' ')
                          .str.replace(',', ' ')
                          .str.replace(r'\s+', ' ', regex=True))

        df['loc_upper'] = df['location'].str.contains(
            'upper', na=False).astype(float)
        df['loc_lower'] = df['location'].str.contains(
            'lower', na=False).astype(float)
        df['loc_isthmus'] = df['location'].str.contains(
            'isthmus', na=False).astype(float)
        df['loc_middle'] = df['location'].str.contains(
            'middle', na=False).astype(float)

        df = df.drop(columns=['location'])
        print(f"  Location: upper={int(df['loc_upper'].sum())}, "
              f"lower={int(df['loc_lower'].sum())}, "
              f"isthmus={int(df['loc_isthmus'].sum())}, "
              f"middle={int(df['loc_middle'].sum())}")

    if li_features:
        df['age_bin'] = (df['age'] < 45).astype(float)
        df['mad_bin'] = (df['MAD'] > 1.0).astype(float)
        df['bmi_bin'] = (df['BMI'] >= 28).astype(float)

        features = ['male', 'age_bin', 'bmi_bin', 'mad_bin',
                     'Multifocality',
                     'loc_upper', 'loc_lower', 'loc_isthmus', 'loc_middle']
        if include_clnm and 'CLNM' in df.columns:
            features.insert(0, 'CLNM')  # Li Model 2: CLNM predictor
            mode_desc = "Li et al. (LLNM+CLNM): 6 binary + 4 location"
        else:
            mode_desc = "Li et al.: 5 binary + 4 location"
    else:
        continuous = ['age', 'BMI', 'MAD']
        binary = ['male', 'Multifocality', 'TgAb', 'TpoAb', 'HT']
        loc_cols = ['loc_upper', 'loc_lower', 'loc_isthmus', 'loc_middle']
        features = continuous + binary + loc_cols
        if include_clnm and 'CLNM' in df.columns:
            features.insert(0, 'CLNM')  # Li'nin LLNM modelinde CLNM predictor
            mode_desc = "KRAL (LLNM+CLNM): 3 cont + 6 binary + 4 location"
        else:
            mode_desc = "KRAL: 3 cont + 5 binary + 4 location"

    available = [c for c in features if c in df.columns]
    print(f"  Features ({len(available)}): {available}")

    X = df[available].astype(np.float64).values

    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            col[miss] = np.nanmedian(col[~miss])

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    return X, y, {
        'name': f'PTC-{target}',
        'source': 'Single center/Thyroid surgery',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': mode_desc,
        'feature_names': available,
        'real_data': True,
        'target': target,
        'li_features': li_features,
    }

def _load_dor(seed=42, comparison=1, li_features=False):
    """
    DOR — Diminished Ovarian Reserve

    comparison=1: Tüm hastalar → clinical pregnancy (n≈900)
    comparison=2: Yumurta alanlar → TE (n≈778)
    comparison=3: ET hastaları → clinical pregnancy (n≈412)

    li_features=True  → Li et al.'ın kullandığı değişkenlerle
    li_features=False → Tüm değişkenlerle
    """
    xlsx_path = os.path.join(os.getcwd(), 'dor.xlsx')
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"dor.xlsx bulunamadı: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_names = xls.sheet_names
    print(f"  Sheet'ler: {sheet_names}")

    dfs = {}
    for i, sheet in enumerate(sheet_names, 1):
        dfs[i] = pd.read_excel(xlsx_path, sheet_name=sheet)
        print(f"  Sheet{i}: {len(dfs[i])} satır")

    # Comparison'a göre sheet ve target
    if comparison == 1:
        df = dfs[1].copy()
    elif comparison == 2:
        df = dfs[2].copy()
    elif comparison == 3:
        df = dfs[3].copy()
    else:
        raise ValueError("comparison 1, 2 veya 3 olmalı")

    df.columns = df.columns.str.strip()

    # ── VİRGÜL ONDALIK ──
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (df[col].astype(str)
                       .str.replace(',', '.', regex=False)
                       .str.replace('，', '.', regex=False))
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > 0.3 * len(df):
                df[col] = converted

    # ── TARGET BUL ──
    y = None
    for col in df.columns:
        col_clean = col.strip().lower()
        if 'clinical pregnancy' in col_clean:
            y = df[col].astype(np.float64).values
            break
        elif 'te(' in col_clean:
            y = df[col].astype(np.float64).values
            break

    if y is None:
        raise ValueError(
            f"Target bulunamadı. Sütunlar: {list(df.columns)}")

    # ── SÜTUN İSİMLERİ ──
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if 'infertility type' in c:
            rename_map[col] = 'infertility_type'
        elif 'years of infertility' in c:
            rename_map[col] = 'infertility_years'
        elif 'previous pregnancy' in c:
            rename_map[col] = 'n_pregnancy'
        elif 'previous delivery' in c:
            rename_map[col] = 'n_delivery'
        elif 'previous abortion' in c:
            rename_map[col] = 'n_abortion'
        elif 'endometrial thickness' in c and 'basal' in c:
            rename_map[col] = 'basal_endometrium'
        elif 'stimulation protocol' in c:
            rename_map[col] = 'protocol'
        elif 'total amount' in c:
            rename_map[col] = 'total_gn'
        elif 'duration of stimulation' in c:
            rename_map[col] = 'stim_days'
        elif 'endometrial thickness' in c and 'hcg' in c:
            rename_map[col] = 'endometrium_hcg'
        elif c == 'e2 on hcg':
            rename_map[col] = 'e2_hcg'
        elif c == 'lh on hcg':
            rename_map[col] = 'lh_hcg'
        elif c in ('p on hcg', 'p on hcg '):
            rename_map[col] = 'p_hcg'
        elif '≧14mm' in c or '≥14mm' in c:
            rename_map[col] = 'n_follicles_14'
        elif 'total oocytes' in c:
            rename_map[col] = 'total_oocytes'
        elif 'transferrable' in c or 'transferable' in c:
            rename_map[col] = 'n_te'
        elif 'good-quality' in c:
            rename_map[col] = 'n_good_embryos'
        elif 'embryo stage' in c:
            rename_map[col] = 'embryo_stage'
        elif 'embryo transferred' in c:
            rename_map[col] = 'n_transferred'
        elif 'type of art' in c:
            rename_map[col] = 'art_type'
        elif c == 'mii':
            rename_map[col] = 'MII'
        elif c == '2pn':
            rename_map[col] = 'twoPN'
        elif 'cycle outcome' in c:
            rename_map[col] = 'cycle_outcome'
        elif 'biochemical' in c:
            rename_map[col] = 'biochem_pregnancy'
        elif 'type of pregnancy' in c:
            rename_map[col] = 'pregnancy_type'
        elif 'age2' == c:
            rename_map[col] = 'age2'
        elif c == 'age':
            rename_map[col] = 'age'
        elif c == 'bmi':
            rename_map[col] = 'BMI'
        elif c == 'fsh':
            rename_map[col] = 'FSH'
        elif c == 'e2':
            rename_map[col] = 'E2'
        elif c == 'p':
            rename_map[col] = 'P'
        elif c == 'lh':
            rename_map[col] = 'LH'
        elif c == 't':
            rename_map[col] = 'T'
        elif c == 'amh':
            rename_map[col] = 'AMH'
        elif c == 'afc':
            rename_map[col] = 'AFC'
    df = df.rename(columns=rename_map)

    # ── MALE AGE (varsa) ──
    if 'age2' in df.columns:
        df['male_age'] = df['age2'].astype(np.float64)

    # ── SIZINTI AT ──
    drop_cols = [
        'KEYCOD', 'age2',
        'n_te', 'n_good_embryos', 'embryo_stage',
        'n_transferred', 'cycle_outcome',
        'biochem_pregnancy', 'abortion', 'pregnancy_type',
        'clinical pregnancy(yes:1,no:0)', 'clinical pregnancy',
        'TE(yes:1,no:0)', 'clinical_pregnancy', 'te',
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                 errors='ignore')

    # ── PROTOKOL ENCODING ──
    if 'protocol' in df.columns:
        if df['protocol'].dtype == object:
            proto_dummies = pd.get_dummies(
                df['protocol'].str.strip(), prefix='proto',
                dummy_na=False)
        else:
            proto_dummies = pd.get_dummies(
                df['protocol'], prefix='proto', dummy_na=False)
        df = pd.concat([df, proto_dummies], axis=1)
        df = df.drop(columns=['protocol'])

    # ── FEATURE SEÇİMİ ──
    if li_features:
        # ═══════════════════════════════════════════
        #  Li et al.'ın KULLANDIĞI değişkenler
        # ═══════════════════════════════════════════
        if comparison == 2:
            # Model 1: TE prediction
            features = ['age', 'FSH', 'AMH', 'AFC']
            mode_desc = "Li Model 1 (TE): age, FSH, AMH, AFC"
        elif comparison == 3:
            # Model 2: Pregnancy in ET
            features = ['infertility_type', 'n_pregnancy',
                        'n_delivery', 'AFC', 'age', 'male_age']
            mode_desc = ("Li Model 2 (ET pregnancy): "
                         "infertility_type, n_pregnancy, "
                         "n_delivery, AFC, age, male_age")
        else:
            raise ValueError(
                "li_features=True sadece comparison=2 veya 3 ile")
    else:
        # ═══════════════════════════════════════════
        #  TÜM değişkenler (orijinal loader)
        # ═══════════════════════════════════════════
        baseline_cont = [
            'age', 'BMI', 'FSH', 'E2', 'P', 'LH', 'T',
            'AMH', 'AFC', 'basal_endometrium',
            'infertility_years', 'n_pregnancy',
            'n_delivery', 'n_abortion',
        ]
        stim_cont = [
            'total_gn', 'stim_days', 'endometrium_hcg',
            'e2_hcg', 'lh_hcg', 'p_hcg',
            'n_follicles_14', 'total_oocytes', 'MII', 'twoPN',
        ]
        binary_cols = ['infertility_type', 'art_type']
        proto_cols = [c for c in df.columns if c.startswith('proto_')]
        features = (baseline_cont + stim_cont
                    + binary_cols + proto_cols)
        mode_desc = f"All {len(features)} features"

    available = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"  ⚠ Eksik sütunlar: {missing}")

    X = df[available].astype(np.float64).values

    # Impute
    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            col[miss] = np.nanmedian(col[~miss])

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    return X, y, {
        'name': f'DOR Comp{comparison}',
        'source': 'Zhengzhou Univ./2011-2019/IVF-ICSI',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': mode_desc,
        'feature_names': available,
        'real_data': True,
        'li_features': li_features,
        'comparison': comparison,
    }

def _load_fic(seed=42):
    """
    FIC — Fluoropyrimidine-Induced Cardiotoxicity
    Kaynak: Guizhou Medical University, CRC patients
    Dosya: fic.xlsx (aynı dizinde, Excel)

    n ≈ 754 hasta
    Target: cardiotoxicity (0/1) — definitive, censored YOK

    Hazırlık:
      - ID sildi, stage düzeltildi (Roma rakamları → sayı)
      - Virgül ondalık düzeltildi
    """
    xlsx_path = os.path.join(os.getcwd(), 'fic.xlsx')
    csv_path = os.path.join(os.getcwd(), 'fic.csv')

    if os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=';', na_values=[
            '', ' ', 'NA', 'NaN', 'N/A'])
    else:
        raise FileNotFoundError(
            f"fic.xlsx veya fic.csv bulunamadı")

    df.columns = df.columns.str.strip()
    df = df.loc[:, df.columns != '']
    df = df.drop(columns=[c for c in df.columns
                          if c.startswith('Unnamed')],
                 errors='ignore')

    # ── VİRGÜL ONDALIK DÜZELTMESİ ──
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (df[col].astype(str)
                       .str.replace(',', '.', regex=False)
                       .str.replace('，', '.', regex=False))
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > 0.5 * len(df):
                df[col] = converted

    # ── SÜTUN İSİMLERİ ──
    rename_map = {}
    for col in df.columns:
        c = col.strip()
        if c == 'clinical manifestation':
            rename_map[col] = 'clinical_manifestation'
        elif c == 'cardiovascular and cerebrovascular':
            rename_map[col] = 'cvc'
        elif c == 'endocrine ' or c == 'endocrine':
            rename_map[col] = 'endocrine'
        elif c == 'targeted drug':
            rename_map[col] = 'targeted_drug'
        elif 'Dosage' in c or 'fluorouracil' in c:
            rename_map[col] = 'fluoro_dosage'
        elif c == 'BUN/Scr':
            rename_map[col] = 'BUN_Scr'
    df = df.rename(columns=rename_map)

    # ── TARGET ──
    target_col = 'cardiotoxicity'
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' bulunamadı. "
                         f"Mevcut: {list(df.columns)}")
    y = df[target_col].astype(np.float64).values

    # ── SIZINTI AT ──
    drop_cols = ['clinical_manifestation', 'CTCAE', target_col]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                 errors='ignore')

    # ── STAGE → ORDINAL ──
    if 'stage' in df.columns:
        # Kullanıcı düzelttiyse sayısal olabilir
        if df['stage'].dtype == object:
            def parse_stage(val):
                if pd.isna(val):
                    return 0
                s = str(val).strip().upper()
                for rom, num in [('IV', '4'), ('III', '3'),
                                 ('II', '2'), ('I', '1')]:
                    s = s.replace(rom, num)
                for ch in s:
                    if ch.isdigit():
                        return float(ch)
                return 0
            df['stage_num'] = df['stage'].apply(parse_stage)
        else:
            df['stage_num'] = df['stage'].astype(float)
        df = df.drop(columns=['stage'])

    # ── SEX → BINARY ──
    if 'Sex' in df.columns:
        df['male'] = (df['Sex'].astype(float) == 1).astype(float)
        df = df.drop(columns=['Sex'])

    # ── FEATURES ──
    continuous_cols = [
        'age', 'BMI', 'SBP', 'DSP',
        'cycle', 'fluoro_dosage',
        'WBC', 'Ne#', 'Ne%', 'LY#', 'LY%', 'M#',
        'EOS#', 'EOS%', 'HGB', 'PLT',
        'ALB', 'AST', 'ALT', 'LDH', 'UA', 'Scr',
        'BUN_Scr', 'K', 'Ca', 'GLU', 'stage_num',
    ]
    binary_cols = [
        'male', 'cvc', 'endocrine', 'drink', 'smoke',
        'chemotherapyregimens', 'targeted_drug',
    ]

    available = [c for c in continuous_cols + binary_cols
                 if c in df.columns]
    X = df[available].astype(np.float64).values

    # Impute
    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            col[miss] = np.nanmedian(col[~miss])

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    return X, y, {
        'name': 'FIC (Cardiotoxicity)',
        'source': 'Guizhou Medical Univ./CRC patients',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': ('Fluoropyrimidine cardiotoxicity, '
                        'sürekli+klinik, censored YOK'),
        'feature_names': available,
        'real_data': True,
        'leakage_removed': [
            'clinical_manifestation (post-outcome)',
            'CTCAE (post-outcome toksisite grading)',
        ],
    }


def _load_breast_cancer(seed=42):
    d = load_breast_cancer()
    return d.data, d.target, {
        'name': 'Breast Cancer', 'source': 'sklearn/UCI',
        'n_raw': d.data.shape[0], 'd_raw': d.data.shape[1],
        'description': 'Meme kanseri (benign vs malignant)',
    }

def _load_wine(seed=42):
    d = load_wine()
    y = (d.target > 0).astype(float)
    return d.data, y, {
        'name': 'Wine', 'source': 'sklearn/UCI',
        'n_raw': d.data.shape[0], 'd_raw': d.data.shape[1],
        'description': 'Şarap (class 0 vs rest)',
    }

def _load_digits(seed=42):
    d = load_digits()
    y = (d.target == 8).astype(float)
    return d.data, y, {
        'name': 'Digits (8 vs rest)', 'source': 'sklearn/UCI',
        'n_raw': d.data.shape[0], 'd_raw': d.data.shape[1],
        'description': 'Rakam tanıma (8 vs diğerleri)',
    }

def _load_iris(seed=42):
    d = load_iris()
    y = (d.target == 2).astype(float)
    return d.data, y, {
        'name': 'Iris (class 2 vs rest)', 'source': 'sklearn/UCI',
        'n_raw': d.data.shape[0], 'd_raw': d.data.shape[1],
        'description': 'Çiçek (virginica vs rest)',
    }


# =====================================================================
# OPENML DATASETS — DÜZELTİLDİ
# =====================================================================

def _openml_load(data_id=None, name=None):
    """OpenML yükleme — parser denemeli."""
    for parser in ['pandas', 'auto']:
        try:
            if data_id is not None:
                return fetch_openml(
                    data_id=data_id, return_X_y=True,
                    as_frame=True, parser=parser)
            else:
                return fetch_openml(
                    name=name, return_X_y=True,
                    as_frame=True, parser=parser)
        except Exception:
            continue
    raise ValueError(f"OpenML yüklenemedi: id={data_id}, name={name}")


def _openml_to_numpy(df_X, y_raw, impute=True):
    """OpenML DataFrame → numpy array."""
    for col in df_X.columns:
        df_X[col] = pd.to_numeric(df_X[col], errors='coerce')

    if impute:
        df_X = pd.DataFrame(
            SimpleImputer(strategy='median').fit_transform(df_X),
            columns=df_X.columns)

    X = df_X.values.astype(np.float64)
    y = y_raw.astype('category').cat.codes.values.astype(np.float64)
    return X, y


def _load_heart(seed=42):
    """
    Heart Disease — UCI Cleveland.
    OpenML data_id=432 ile yüklenir.
    Kontrol: n≈303, d≈13 olmalı.
    """
    df_X, y_raw = _openml_load(data_id=432)

    if len(df_X) < 100 or len(df_X.columns) > 20:
        raise ValueError(
            f"Heart: beklenmeyen boyut n={len(df_X)}, "
            f"d={len(df_X.columns)}. "
            "OpenML farklı data döndürdü.")

    X, y = _openml_to_numpy(df_X, y_raw, impute=True)

    if set(np.unique(y).astype(int)) - {0, 1}:
        y = (y > 0).astype(float)

    return X, y, {
        'name': 'Heart Disease', 'source': 'UCI/OpenML(id=432)',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': 'Kalp hastalığı (Cleveland)',
        'imputed': True,
        'feature_names': list(df_X.columns),
    }


def _load_diabetes(seed=42):
    """Pima Indians Diabetes — OpenML id=37."""
    df_X, y_raw = _openml_load(data_id=37)

    zero_cols = ['plas', 'pres', 'skin', 'insu', 'mass']
    for col in zero_cols:
        if col in df_X.columns:
            df_X[col] = pd.to_numeric(
                df_X[col], errors='coerce').replace(0, np.nan)

    X, y = _openml_to_numpy(df_X, y_raw, impute=True)

    return X, y, {
        'name': 'Pima Diabetes', 'source': 'OpenML(id=37)',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': 'Diyabet (Pima Indian)',
        'imputed': True,
        'imputed_columns': zero_cols,
    }


def _load_ionosphere(seed=42):
    """
    Ionosphere — UCI.
    OpenML data_id=59 ile yüklenir.
    Kontrol: n≈351, d≈34 olmalı.
    """
    df_X, y_raw = _openml_load(data_id=59)

    if len(df_X) < 100 or len(df_X.columns) < 10:
        raise ValueError(
            f"Ionosphere: beklenmeyen boyut n={len(df_X)}, "
            f"d={len(df_X.columns)}. "
            "OpenML farklı data döndürdü.")

    X, y = _openml_to_numpy(df_X, y_raw, impute=True)

    return X, y, {
        'name': 'Ionosphere', 'source': 'UCI/OpenML(id=59)',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': 'İyonosfer radar sinyali',
    }


def _load_australian(seed=42):
    """Australian Credit — OpenML id=40981."""
    df_X, y_raw = _openml_load(data_id=40981)
    X, y = _openml_to_numpy(df_X, y_raw, impute=True)

    return X, y, {
        'name': 'Australian Credit', 'source': 'OpenML(id=40981)',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': 'Kredi onay (Avustralya)',
    }


# =====================================================================
# SENTETİK DATASETLER
# =====================================================================

def _load_sim(seed=42):
    """
    SIM — Makale ile uyumlu sentetik nonlinear veri.

    DGP (orijinal):
        bmi_risk     = 0.8 * ((BMI - 24)^2) / 10
        glucose_risk = threshold nonlinearity
        age_smoke    = 0.04 * Age * Smoke      ← interaction
        bp_risk      = piecewise linear + quadratic

    Ground truth: β lineer katsayılar approximately,
    ama nonlinear terimler dominant.
    """
    rng = np.random.default_rng(seed)
    n = 5000

    Age = rng.normal(55, 12, n)
    BMI = rng.normal(27, 5, n)
    BP = rng.normal(130, 18, n)
    Glucose = rng.normal(100, 25, n)
    Smoke = rng.binomial(1, 0.25, n)

    X = np.column_stack([Age, BMI, BP, Glucose, Smoke])

    bmi_risk = 0.8 * ((BMI - 24) ** 2) / 10
    glucose_risk = np.where(Glucose > 110,
                            0.05 * np.exp((Glucose - 110) / 20),
                            0.01 * (Glucose - 90))
    age_smoke = 0.04 * Age * Smoke
    bp_risk = (0.02 * (BP - 120) +
               0.0005 * (BP - 140) ** 2 * (BP > 140))

    eta = (-4.5 + bmi_risk + glucose_risk + age_smoke
           + bp_risk + 0.3 * Smoke + 0.03 * Age)

    prob = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)

    feature_names = ['Age', 'BMI', 'BP', 'Glucose', 'Smoke']

    return X, y, {
        'name': 'SIM (Synthetic)', 'source': 'Generated',
        'n_raw': n, 'd_raw': 5,
        'description': ('Nonlinear: U-BMI, threshold-glucose, '
                        'age×smoke'),
        'feature_names': feature_names,
        # ── Ground truth for Table 2 (β recovery) ──
        'true_beta': {
            'beta': np.array([0.03, 0.0, 0.02, 0.0, 0.30]),
            'intercept': -4.5,
            'names': feature_names,
            'note': ('Approximate linear coefficients. '
                     'True DGP is nonlinear — BMI has U-shape, '
                     'Glucose has threshold, '
                     'Age×Smoke interaction.'),
        },
        # ── Ground truth for Table 3 (interactions) ──
        'true_interactions': [
            ('Age', 'Smoke'),   # 0.04 * Age * Smoke
        ],
        'true_nonlinear': [
            'BMI² (U-shape: 0.8*(BMI-24)²/10)',
            'Glucose (threshold at 110)',
            'BP² (piecewise quadratic above 140)',
        ],
        'dgp_formula': (
            'η = -4.5 + 0.8·(BMI-24)²/10 '
            '+ threshold(Glucose,110) '
            '+ 0.04·Age·Smoke + piecewise(BP) '
            '+ 0.3·Smoke + 0.03·Age'),
    }


def _load_linear(seed=42):
    """
    Sentetik doğrusal veri.
    logit = Xw + b, 3 irrelevant features.
    Gerçek β biliniyor → karşılaştırma için.
    """
    rng = np.random.default_rng(seed)
    n = 5000
    d = 8

    X = rng.normal(size=(n, d))

    # Gerçek katsayılar
    w = np.array([0.8, -1.2, 0.5, 0.0, 0.0, 0.3, -0.6, 0.0])
    b = 0.2
    noise = rng.normal(scale=0.5, size=n)

    eta = X @ w + b + noise
    prob = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)

    feature_names = [f'x{i+1}' for i in range(d)]

    return X, y, {
        'name': 'LINEAR (Synthetic)', 'source': 'Generated',
        'n_raw': n, 'd_raw': d,
        'description': 'logit = Xw + b, 3 irrelevant features',
        'feature_names': feature_names,
        'true_weights': w.tolist(),
        'true_bias': b,
        # ── Ground truth for Table 2 (β recovery) ──
        'true_beta': {
            'beta': w,
            'intercept': b,
            'names': feature_names,
            'note': ('Exact linear DGP. x4, x5, x8 are '
                     'irrelevant (β=0). Noise σ=0.5.'),
        },
        # ── Ground truth for Table 3 (interactions) ──
        'true_interactions': [],  # no interactions
        'true_nonlinear': [],     # no nonlinearity
        'dgp_formula': (
            'η = 0.8·x1 - 1.2·x2 + 0.5·x3 '
            '+ 0.3·x6 - 0.6·x7 + 0.2 + ε, ε~N(0,0.5)'),
    }


import os
import urllib.request
import zipfile
import io

try:
    _DATA_DIR = os.environ.get(
        'KALE_DATA_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'data_cache'))
except NameError:
    _DATA_DIR = os.environ.get('KALE_DATA_DIR', 'data_cache')


def _fetch_zip(url, dirname):
    """ZIP indir → çıkar (önbellekli)."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    target = os.path.join(_DATA_DIR, dirname)
    if os.path.isdir(target) and any(
            f.endswith('.csv') for f in os.listdir(target)):
        return target
    zip_path = os.path.join(_DATA_DIR, dirname + '.zip')
    print(f"  ↓ {dirname} indiriliyor ({url[:60]}...)")
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target)
        os.remove(zip_path)
        print(f"  ✓ {dirname} hazır")
    except Exception as e:
        raise ValueError(
            f"İndirme başarısız: {url}\n  {e}\n"
            f"  Manuel indirip {target}/ dizinine çıkarın.")
    return target


def _find_csv(directory, pattern):
    """Dizinde CSV dosyası ara."""
    for root, _, files in os.walk(directory):
        for f in files:
            if pattern in f.lower() and f.endswith('.csv'):
                return os.path.join(root, f)
    return None


# =====================================================================
# 1. UCI DIABETES 130-US HOSPITALS  (gerçek, ~101K, otomatik indirir)
# =====================================================================
def _load_support2(seed=42):
    """
    SUPPORT2 — GERÇEK klinik mortalite veri seti.
    SIZINTI TEMİZ — sadece kabul anı bilgisi.

    Kaynak  : UCI (id=880)
    Boyut   : ~9,105 hasta
    Hedef   : Hastane mortalitesi (1) vs taburcu (0)
    """
    csv_path = os.path.join(os.getcwd(), 'support2.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"support2.csv bulunamadı: {csv_path}\n"
            f"Çalışma dizinine support2.csv koy.")

    df = pd.read_csv(csv_path, na_values=['?'])

    y = df['hospdead'].astype(np.float64).values

    # ── SIZINTI YOK: sadece kabul anı öznitelikleri ──
    num_cols = [
        'age',          # demografik
        'num.co',       # komorbidite sayısı
        'sps',          # SUPPORT fizyoloji skoru (kabul anı)
        'aps',          # Apache III skoru (kabul anı)
        'diabetes',     # komorbidite
        'dementia',     # komorbidite
        'meanbp',       # kan basıncı (kabul anı)
        'wblc',         # lökosit (kabul anı)
        'hrt',          # nabız (kabul anı)
        'resp',         # solunum (kabul anı)
        'temp',         # sıcaklık (kabul anı)
        'alb',          # albumin (kabul anı lab)
        'bili',         # bilirubin (kabul anı lab)
        'crea',         # kreatinin (kabul anı lab)
        'sod',          # sodyum (kabul anı lab)
        'ph',           # pH (kabul anı lab)
        'glucose',      # glikoz (kabul anı lab)
        'bun',          # BUN (kabul anı lab)
        'urine',        # idrar çıkışı (kabul anı)
        'adlp',         # ADL hasta
        'adls',         # ADL vekil
    ]

    # ── SIZINTI VAR → ATILDI ──
    # 'd.time'    → ölüm/takip süresi (HEDFI TAŞIYOR)
    # 'prg2m'     → doktor 2 ay survival tahmini (HEDFI TAŞIYOR)
    # 'prg6m'     → doktor 6 ay survival tahmini (HEDFI TAŞIYOR)
    # 'surv2m'    → model 2 ay survival tahmini (HEDFI TAŞIYOR)
    # 'surv6m'    → model 6 ay survival tahmini (HEDFI TAŞIYOR)
    # 'slos'      → kalış süresi (post-admission, dolaylı sızıntı)
    # 'hday'      → kayıt günü (post-admission)

    df['sex_num'] = (df['sex'] == 'male').astype(float)

    race_map = {'white': 0, 'black': 1, 'hispanic': 2,
                'asian': 3, 'other': 4}
    df['race_num'] = df['race'].map(race_map).fillna(0)

    ca_map = {'no': 0, 'yes': 1, 'metastatic': 2}
    df['ca_num'] = df['ca'].map(ca_map).fillna(0)

    cat_cols = ['sex_num', 'race_num', 'ca_num']

    available = [c for c in num_cols if c in df.columns]
    available_cat = [c for c in cat_cols if c in df.columns]

    X = df[available + available_cat].astype(np.float64).values

    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            col[miss] = np.nanmedian(col)

    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    actual_names = available + ['Sex', 'Race', 'Cancer']

    return X, y, {
        'name': 'SUPPORT2',
        'source': 'UCI(id=880)/5 hospitals/1989-1994',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': ('Klinik mortalite, SIZINTI TEMİZ, '
                        'sadece kabul anı öznitelikleri'),
        'feature_names': actual_names,
        'real_data': True,
        'leakage_removed': [
            'd.time (ölüm süresi)',
            'prg2m/prg6m (doktor tahmini)',
            'surv2m/surv6m (model tahmini)',
            'slos (kalış süresi)',
            'hday (kayıt günü)',
        ],
    }
def _load_flchain(seed=42):
    """
    FLCHAIN — Free Light Chain mortality.
    Kaynak: R survival paketi, ~7,874 hasta

    Hedef: death (orijinal)
    Min follow-up filtresi: death=0 VE futime<365 gün → censored → DROP

    Sızıntı temizliği:
      - futime    ATILDI
      - chapter   ATILDI
      - flc.grp   ATILDI
      - sample.yr ATILDI
    """
    url = ("https://raw.githubusercontent.com/vincentarelbundock/"
           "Rdatasets/master/csv/survival/flchain.csv")
    df = pd.read_csv(url)
    df = df.drop(columns=["Unnamed: 0", "rownames"],
                 errors="ignore")

    MIN_FOLLOWUP = 365

    # ── Censored DROPLE ──
    mask_censored = (df['death'] == 0) & (df['futime'] < MIN_FOLLOWUP)
    n_dropped = mask_censored.sum()
    df = df[~mask_censored].copy()

    # ── Hedef: orijinal death ──
    y = df['death'].astype(np.float64).values

    # ── SIZINTI AT ──
    drop_cols = ['death', 'futime', 'chapter', 'flc.grp', 'sample.yr']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                 errors='ignore')

    # ── FEATURES ──
    df["sex_num"] = (df["sex"] == "M").astype(float)
    num_cols = ["age", "kappa", "lambda", "creatinine", "mgus"]
    available = [c for c in num_cols if c in df.columns]
    X = df[available + ["sex_num"]].astype(np.float64).values

    # Impute
    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            col[miss] = np.nanmedian(col[~miss])

    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    feature_names = available + ["Sex"]

    return X, y, {
        'name': 'FLCHAIN (Mortality)',
        'source': 'R survival package',
        'n_raw': len(y),
        'd_raw': X.shape[1],
        'description': (f'Mortality, censored drop={n_dropped}, '
                        f'pos_rate={y.mean():.1%}'),
        'feature_names': feature_names,
        'real_data': True,
        'leakage_removed': [
            'futime',
            'chapter',
            'flc.grp',
            'sample.yr',
        ],
    }

def _load_metabric(seed=42):
    """
    METABRIC — Meme kanseri mortalite.
    Kaynak: pycox, ~1,904 hasta
    Hedef: event (0/1)
    SIZINTI TEMİZ: tedavi indeksleri (x7, x8) atıldı.
    """
    from pycox.datasets import metabric
    df = metabric.read_df()

    y = df["event"].astype(np.float64).values

    # x7 = radiotherapy, x8 = chemotherapy → at (confounding)
    drop_cols = ["duration", "event", "x7", "x8"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    real_names = [
        "age", "menopause", "tumor_size", "inv_nodes",
        "nodes", "ER_status", "grade",
    ]

    assert len(feature_cols) == len(real_names), (
        f"Beklenen {len(real_names)} sütun, "
        f"bulunan {len(feature_cols)}: {feature_cols}")

    df_feat = df[feature_cols].copy()
    df_feat.columns = real_names

    for col in real_names:
        df_feat[col] = df_feat[col].fillna(df_feat[col].median())

    X = df_feat.values.astype(np.float64)

    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    return X, y, {
        'name': 'METABRIC',
        'source': 'pycox/1,904 meme kanseri hasta',
        'n_raw': len(y), 'd_raw': X.shape[1],
        'description': 'Meme kanseri mortalite, patoloji+demografik',
        'feature_names': real_names,
        'real_data': True,
        'leakage_removed': [
            'x7=radiotherapy (tedavi, confounding)',
            'x8=chemotherapy (tedavi, confounding)',
        ],
    }
def _load_messidor(seed=42):
    """
    Messidor Diabetic Retinopathy — ARFF dosyasından.
    Dosya: messidor_features.arff (aynı dizinde)
    Sütun eşleme (UCI dökümantasyonu):
      0  = quality
      1  = pre_screening       → SIZINTI
      2..7  = ma1..ma6
      8..15 = exudate1..exudate8
      16 = macula_opticdisc_distance
      17 = opticdisc_diameter
      18 = am_fm_classification → SIZINTI
      Class = target
    """
    arff_path = os.path.join(os.getcwd(), 'messidor_features.arff')
    if not os.path.exists(arff_path):
        raise FileNotFoundError(
            f"messidor_features.arff bulunamadı: {arff_path}")

    # ARFF parse
    attributes = []
    data_lines = []
    in_data = False

    with open(arff_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@attribute'):
                parts = line.split(None, 2)
                attributes.append(parts[1].strip("'\""))
            elif line.lower().startswith('@data'):
                in_data = True
            elif in_data:
                vals = [v.strip().strip("'\"") for v in line.split(',')]
                data_lines.append(vals)

    df = pd.DataFrame(data_lines, columns=attributes)

    # Sayısala çevir
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Target
    target_col = 'Class' if 'Class' in df.columns else 'class'
    y = df[target_col].astype(np.float64).values

    # ── SÜTUN İSİMLERİ (sayısal → anlamlı) ──
    col_map = {
        '0': 'quality',
        '1': 'pre_screening',        # SIZINTI
        '2': 'ma1', '3': 'ma2', '4': 'ma3',
        '5': 'ma4', '6': 'ma5', '7': 'ma6',
        '8': 'exudate1', '9': 'exudate2',
        '10': 'exudate3', '11': 'exudate4',
        '12': 'exudate5', '13': 'exudate6',
        '14': 'exudate7', '15': 'exudate8',
        '16': 'macula_od_dist',
        '17': 'opticdisc_diam',
        '18': 'am_fm_class',         # SIZINTI
    }
    df = df.rename(columns=col_map)

    # ── SIZINTI AT ──
    df = df.drop(
        columns=[c for c in ['pre_screening', 'am_fm_class',
                              target_col] if c in df.columns],
        errors='ignore')

    # ── MULTİKOLİNEARİTE AZALT ──
    drop_multi = [f'ma{i}' for i in range(2, 6)] + \
                 [f'exudate{i}' for i in range(2, 8)]
    df = df.drop(
        columns=[c for c in drop_multi if c in df.columns],
        errors='ignore')

    # ── 7 TEMİZ FEATURE ──
    feature_cols = [
        'quality', 'ma1', 'ma6',
        'exudate1', 'exudate8',
        'macula_od_dist', 'opticdisc_diam',
    ]
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].astype(np.float64).values

    # Impute
    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            col[miss] = np.nanmedian(col[~miss])

    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    return X, y, {
        'name': 'Messidor DR',
        'source': 'UCI(329)/Messidor fundus images',
        'n_raw': len(y),
        'd_raw': X.shape[1],
        'description': ('Diabetic retinopathy tarama, '
                        'sızıntı-temiz, multikolinearite-azaltılmış'),
        'feature_names': available,
        'real_data': True,
        'leakage_removed': [
            'pre_screening (önceki model çıktısı)',
            'am_fm_classification (AM/FM model çıktısı)',
        ],
        'multicollinearity_reduced': [
            'ma2-5 atıldı → ma1(α=0.5) + ma6(α=1.0)',
            'exudate2-7 atıldı → exudate1 + exudate8',
        ],
    }
def _load_vtc(seed=42):
    """
    VTC — Vancomycin Trough Concentration & ICU Mortality
    Kaynak: eICU-CRD / Hou et al. (2021)
    Dosya: vtc.csv (aynı dizinde)

    n ≈ 3,603 hasta
    Target: unitdischargestatus (ICU ölümü)
    Ana predictor: Mean VTC (sürekli!)

    SIZINTI TEMİZ:
      - hospitaldischargestatus ATILDI (alternatif target)
      - dosage_sum ATILDI (post-hoc ilaç bilgisi)
      - drugstartoffset ATILDI (zaman bilgisi)
      - drugstopoffset ATILDI (zaman bilgisi)
      - frequency ATILDI (dozaj sıklığı)
      - patientunitstayid ATILDI (ID)
    """
    csv_path = os.path.join(os.getcwd(), 'vtc.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"vtc.csv bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path, na_values=[' ', '', 'NA', 'NaN'])

    # ── TARGET ──
    # unitdischargestatus: ICU çıkış durumu
    # 0=ALIVE, 1=EXPIRED veya string olabilir
    target_col = 'unitdischargestatus'
    if target_col not in df.columns:
        raise ValueError(f"Target sütunu '{target_col}' bulunamadı")

    # String ise çevir
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].str.strip().str.upper()
        df[target_col] = df[target_col].map({
            'EXPIRED': 1, 'ALIVE': 0,
            'DEAD': 1, 'LIVING': 0,
        })
    y = df[target_col].astype(np.float64).values

    # ── SIZINTI AT ──
    drop_cols = [
        'patientunitstayid',
        'hospitaldischargestatus',  # alternatif target
        'dosage_sum',               # post-hoc ilaç bilgisi
        'drugstartoffset',          # zaman bilgisi
        'drugstopoffset',           # zaman bilgisi
        'frequency',                # dozaj sıklığı
        target_col,                 # target
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                 errors='ignore')

    # ── GENDER KODLAMA ──
    if 'gender' in df.columns:
        df['male'] = (df['gender'].str.strip().str.upper()
                      .isin(['MALE', 'M'])).astype(float)
        df = df.drop(columns=['gender'])

    # ── ETHNICITY KODLAMA ──
    if 'ethnicity' in df.columns:
        df['caucasian'] = (df['ethnicity'].str.strip().str.upper()
                           .str.contains('CAUCASIAN|WHITE')).astype(float)
        df = df.drop(columns=['ethnicity'])

    # ── SÜREKLİ FEATURES (KRAL ∂η/∂x için ideal) ──
    continuous_cols = [
        'Mean',           # Ana predictor: Mean VTC (sürekli)
        'age',            # Yaş (sürekli)
        'BMI',            # BMI (sürekli)
        'apachescore',    # APACHE IV skoru (sürekli)
        'creatinineavg',  # Ortalama kreatinin (sürekli)
        'CCl',            # CrCl - Cockcroft-Gault (sürekli)
        'N',              # VTC ölçüm sayısı (sürekli)
    ]

    # ── BINARY FEATURES (confounding düzeltmesi) ──
    binary_cols = [
        'ventilation',
        'dialysis',
        'vasopressor',
        'vasodepressor',
        'tumour',
        'hepatic_failure',
        'copd',
        'heart_failure',
        'diabetes',
        'Gastrointestinal_Bleed',
        'pancreatitis',
        'burns',
        'Pneumonia',
        'sepsis',
        'renal_failure',
        'male',
        'caucasian',
    ]

    # Mean sütununu vtc_mean olarak yeniden adlandır
    if 'Mean' in df.columns:
        df = df.rename(columns={'Mean': 'vtc_mean'})

    continuous_cols = ['vtc_mean' if c == 'Mean' else c
                       for c in continuous_cols]

    available_cont = [c for c in continuous_cols if c in df.columns]
    available_bin = [c for c in binary_cols if c in df.columns]
    all_features = available_cont + available_bin

    X = df[all_features].astype(np.float64).values

    # Impute
    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            col[miss] = np.nanmedian(col[~miss])

    # Temizlik
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    # Hedef NaN kontrolü
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    feature_names = available_cont + available_bin

    return X, y, {
        'name': 'VTC (Vancomycin)',
        'source': 'eICU-CRD/3,603 ICU patients',
        'n_raw': len(y),
        'd_raw': X.shape[1],
        'description': ('Vancomycin trough → ICU mortalite, '
                        'sürekli VTC/yaş/BMI/CrCl, censored YOK'),
        'feature_names': feature_names,
        'real_data': True,
        'continuous_predictors': available_cont,
        'binary_predictors': available_bin,
        'primary_predictor': 'vtc_mean',
        'leakage_removed': [
            'hospitaldischargestatus (alternatif target)',
            'dosage_sum (post-hoc ilaç)',
            'drugstartoffset/drugstopoffset (zaman)',
            'frequency (dozaj sıklığı)',
        ],
        'expected_nlm_pattern': 'U-shape (düşük ve yüksek VTC → ölüm)',
    }

# =====================================================================
# YARDIMCI
# =====================================================================

def load_all(normalize=True, seed=42, datasets=None):
    """Tüm veya seçili veri setlerini yükle."""
    all_names = [
        'breast_cancer', 'wine', 'digits', 'iris',
        'heart', 'diabetes', 'ionosphere', 'australian',
        'sim', 'linear','messidor'
    ]
    if datasets:
        all_names = [n for n in all_names if n in datasets]

    results = []
    for name in all_names:
        try:
            X, y, meta = load_dataset(name, normalize=normalize,
                                       seed=seed)
            results.append((X, y, meta))
            print(f"  ✓ {meta['name']:<25} "
                  f"n={meta['n_final']:>5}, d={meta['d_final']:>3}, "
                  f"pos={meta['pos_rate']:.1%}")
        except Exception as e:
            print(f"  ✗ {name:<25} FAILED: {e}")
    return results


def dataset_summary():
    """Özet tablo."""
    print(f"\n{'=' * 85}")
    print(f"  KALE Benchmark Datasets")
    print(f"{'=' * 85}")
    print(f"  {'Name':<25} {'n':>6} {'d':>4} {'Pos%':>7} "
          f"{'C0':>6} {'C1':>6} {'Source':<15}")
    print(f"  {'-' * 85}")

    data = load_all(normalize=False)
    for X, y, meta in data:
        print(f"  {meta['name']:<25} {meta['n_final']:>6} "
              f"{meta['d_final']:>4} {meta['pos_rate']:>7.1%} "
              f"{meta['class_0']:>6} {meta['class_1']:>6} "
              f"{meta['source']:<15}")
    print(f"{'=' * 85}")


if __name__ == '__main__':
    print("╔" + "═" * 75 + "╗")
    print("║  KALE — Dataset Loader (Düzeltildi)                                    ║")
    print("╚" + "═" * 75 + "╝")

    print("\n  [1] All datasets (normalized):")
    load_all(normalize=True)

    dataset_summary()
