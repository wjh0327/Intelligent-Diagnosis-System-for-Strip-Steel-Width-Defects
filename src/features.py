# -*- coding: utf-8 -*-
"""
v3 特征提取模块
================
输入: data/width.csv —— STRIPNO + 6个工艺参数 + RMDATASEQ/FMDATASEQ/DCDATASEQ 三段宽度曲线 + label

产出两组特征:
1. flags6   : 忠实移植 te.py 的 6 个 0/1 规则特征 (is_FM/is_RM/is_DM/is_HT/is_WS/is_NA)
              —— 仅用于对照实验，验证"原方案 = 分类器重学规则"的假设
2. cont     : 新的连续免阈值曲线特征。每个轧段约20维(偏差/头尾跌落/自适应波谷/中段不均匀度等)
              + 8维跨段特征，全部有物理含义，无需按产线手调阈值
              —— 波谷检测的 prominence 由信号自身噪声水平(MAD)自适应，替代写死的 1/5

曲线清洗与 te.py 对齐: 去0、掐头去尾各3点；但不再做 ±30mm 硬删除，
改为记录离群点占比(frac_outlier)作为特征 + 统计时截断到 ±40mm 保证稳健。
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

TRIM = 3           # 与 te.py 一致: 掐头去尾各3点
CLIP_MM = 40.0     # 稳健统计截断半径
OUTLIER_MM = 30.0  # 离群点判定半径(占比作为特征)
SMOOTH_WIN = 5     # 曲线平滑窗口

# te.py config.yaml 的原值，仅供 flags6 移植使用
CFG = dict(REMAINDER=8, LOCAL_RIGHT=1, HEADROTATION=0.2, TAILROTATION=0.8,
           AIM_COL_PLUS=2, SWELL_THRESHOLD=2, NARROW_THRESHOLD=2,
           LOCAL_THRESHOLD=15, PROMINENCE=1, WIDTH=1, PROMINENCE_RM=5,
           WIDTH_RM=1, FIX=2, DH=20)

BASE_COLS = ['FMWTARGETHOT', 'RDWTARGETTOTAL', 'PDIWIDTHTOL',
             'FMWIDTHACTHOT', 'RMWIDTHACTHOT', 'FMWTARGETCOL']
FLAG_COLS = ['is_FM', 'is_RM', 'is_DM', 'is_HT', 'is_WS', 'is_NA']

STAGE_NAMES = ['n_points', 'frac_outlier', 'mean_dev_mm', 'mean_dev_rel', 'std_rel',
               'iqr_rel', 'range_rel', 'head_dev_mm', 'tail_dev_mm', 'mid_dev_mm',
               'head_drop_rel', 'tail_drop_rel', 'drift_rel', 'n_valleys',
               'vmin_dev_mm', 'vmin_pos', 'valley_depth_max_mm', 'valley_depth_base_mm',
               'mid_nonuniform_rel', 'noise_rel']

CROSS_NAMES = ['fm_act_hot_dev_mm', 'fm_act_col_dev_mm', 'fm_margin_lower_mm',
               'fm_mean_margin_lower_mm', 'dc_margin_lower_mm', 'rm_act_dev_mm',
               'ratio_fm_rm', 'ratio_dc_fm', 'dc_fm_mean_diff_mm',
               'fm_head_tail_asym_mm', 'rm_fm_dev_diff_mm']

CONT_COLS = ([f'{s}_{n}' for s in ('RM', 'FM', 'DC') for n in STAGE_NAMES]
             + CROSS_NAMES)
ALL_CONT = BASE_COLS + CONT_COLS          # 连续特征 + 工艺参数上下文
ALL_FEATS = BASE_COLS + FLAG_COLS + CONT_COLS
FEAT12 = BASE_COLS + FLAG_COLS          # 主模型(混合CNN)特征分支的12维输入  # 全量


# ---------------------------------------------------------------- 曲线解析与清洗

def parse_curve(s):
    """逗号分隔字符串 -> float数组，去NaN"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.array([])
    try:
        v = np.array([float(t) for t in str(s).split(',') if t.strip() != ''])
    except ValueError:
        return np.array([])
    return v[~np.isnan(v)]


def clean_curve(x):
    """去0、掐头去尾各3点（与te.py对齐）"""
    x = x[x != 0]
    if len(x) > 2 * TRIM:
        x = x[TRIM:-TRIM]
    return x


def moving_average(x, w=SMOOTH_WIN):
    if len(x) < w:
        return x.astype(float).copy()
    return np.convolve(x, np.ones(w) / w, mode='same')

def wilson_ci(k, n, z=1.96):
    """按类召回率的95%Wilson置信区间（k=正确数, n=样本数），供各实验脚本共用"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


# ---------------------------------------------------------------- 连续段特征

def stage_features(x_raw, target, prefix):
    """单轧段连续特征。target 为该段目标宽度(热值)，x_raw 为原始曲线。"""
    keys = [f'{prefix}_{n}' for n in STAGE_NAMES]
    x = clean_curve(parse_curve(x_raw))
    f = dict.fromkeys(keys, np.nan)
    if len(x) < 10 or not target or target <= 0:
        return f

    T = float(target)
    n_raw = len(x)
    frac_out = float(np.mean(np.abs(x - T) > OUTLIER_MM))
    xs = np.clip(x, T - CLIP_MM, T + CLIP_MM).astype(float)
    n = len(xs)
    mean = xs.mean()

    f[f'{prefix}_n_points'] = np.log1p(n_raw)
    f[f'{prefix}_frac_outlier'] = frac_out
    f[f'{prefix}_mean_dev_mm'] = mean - T
    f[f'{prefix}_mean_dev_rel'] = (mean - T) / T
    f[f'{prefix}_std_rel'] = xs.std() / T
    p5, p25, p75, p95 = np.percentile(xs, [5, 25, 75, 95])
    f[f'{prefix}_iqr_rel'] = (p75 - p25) / T
    f[f'{prefix}_range_rel'] = (p95 - p5) / T

    k = max(3, int(n * 0.1))
    head, tail = xs[:k], xs[-k:]
    mid = xs[k:n - k]
    mid_mean = mid.mean() if len(mid) else mean
    f[f'{prefix}_head_dev_mm'] = head.mean() - T
    f[f'{prefix}_tail_dev_mm'] = tail.mean() - T
    f[f'{prefix}_mid_dev_mm'] = mid_mean - T
    f[f'{prefix}_head_drop_rel'] = (mid_mean - head.mean()) / T   # >0: 头部窄于中段
    f[f'{prefix}_tail_drop_rel'] = (mid_mean - tail.mean()) / T   # >0: 尾部窄于中段

    idx = np.arange(n)
    slope = np.polyfit(idx, xs, 1)[0]
    f[f'{prefix}_drift_rel'] = slope * n / T                      # 全长总漂移

    # 自适应波谷检测: prominence 随信号噪声缩放，替代写死阈值
    sm = moving_average(xs)
    diff = np.diff(xs)
    noise = np.median(np.abs(diff - np.median(diff))) * 1.4826 if len(diff) else 0.0
    prom = max(1.0, 2.0 * noise)
    valleys, _ = find_peaks(-sm, prominence=prom)
    f[f'{prefix}_n_valleys'] = len(valleys)
    if len(valleys):
        vmin = sm[valleys].min()
        f[f'{prefix}_vmin_dev_mm'] = vmin - T
        f[f'{prefix}_vmin_pos'] = valleys[np.argmin(sm[valleys])] / n
        f[f'{prefix}_valley_depth_max_mm'] = max(0.0, T - vmin)
        f[f'{prefix}_valley_depth_base_mm'] = float(np.median(sm) - vmin)
    else:
        f[f'{prefix}_vmin_dev_mm'] = sm.min() - T
        f[f'{prefix}_vmin_pos'] = float(np.argmin(sm)) / n
        f[f'{prefix}_valley_depth_max_mm'] = 0.0
        f[f'{prefix}_valley_depth_base_mm'] = 0.0

    if len(mid) > 5:
        mp5, mp95 = np.percentile(mid, [5, 95])
        f[f'{prefix}_mid_nonuniform_rel'] = (mp95 - mp5) / T
    else:
        f[f'{prefix}_mid_nonuniform_rel'] = (p95 - p5) / T
    f[f'{prefix}_noise_rel'] = noise / T
    return f


def cross_features(row, curves):
    """跨段/冷热态换算特征。curves = dict(RM=arr, FM=arr, DC=arr) 已 clean。"""
    hot_cold_ratio = row['FMWTARGETHOT'] / row['FMWTARGETCOL']
    fm_act_col = row['FMWIDTHACTHOT'] / hot_cold_ratio     # 热值折算冷值(与te.py一致)
    fm_mean_curve = curves['FM'].mean() if len(curves['FM']) else np.nan
    dc_mean_curve = curves['DC'].mean() if len(curves['DC']) else np.nan
    # 冷态下限(不含te.py的FIX=2魔法常数, 树模型可自行学出等效阈值)
    lower_cold = row['FMWTARGETCOL'] - row['PDIWIDTHTOL']

    f = {
        'fm_act_hot_dev_mm': row['FMWIDTHACTHOT'] - row['FMWTARGETHOT'],  # 精轧热值偏差
        'fm_act_col_dev_mm': fm_act_col - row['FMWTARGETCOL'],          # 精轧冷值偏差
        'fm_margin_lower_mm': fm_act_col - lower_cold,                  # 标量口径: 高于冷态下限余量
        'fm_mean_margin_lower_mm': fm_mean_curve - lower_cold,          # 曲线口径: 对应te.py is_NA条件1
        'dc_margin_lower_mm': dc_mean_curve - lower_cold,               # 跨基准: 对应te.py is_NA条件2
        'rm_act_dev_mm': row['RMWIDTHACTHOT'] - row['RDWTARGETTOTAL'],  # 粗轧热值偏差
        'ratio_fm_rm': row['FMWIDTHACTHOT'] / row['RMWIDTHACTHOT'],
        'ratio_dc_fm': dc_mean_curve / fm_mean_curve if fm_mean_curve else np.nan,
        'dc_fm_mean_diff_mm': dc_mean_curve - fm_mean_curve,            # 卷取相对精轧的收窄
        'fm_head_tail_asym_mm': (curves['FM'][:max(3, int(len(curves['FM']) * 0.1))].mean()
                                 - curves['FM'][-max(3, int(len(curves['FM']) * 0.1)):].mean()
                                 ) if len(curves['FM']) > 10 else np.nan,
        'rm_fm_dev_diff_mm': (row['RMWIDTHACTHOT'] - row['RDWTARGETTOTAL'])
                             - (row['FMWIDTHACTHOT'] - row['FMWTARGETHOT']),
    }
    return f


# ---------------------------------------------------------------- te.py 规则特征(忠实移植)

def _seq_te(s, lo=None, hi=None, trim_first=False):
    """按 te.py 各段的实际顺序解析: RM/DC 先滤0再掐头尾, FM 先掐头尾再滤0"""
    try:
        lst = [float(t) for t in str(s).split(',') if t.strip() != '']
    except (ValueError, AttributeError):
        return []
    if not trim_first:
        lst = [v for v in lst if v != 0]
    if len(lst) > 2 * TRIM:
        del lst[0:TRIM]
        del lst[-TRIM:]
    if trim_first:
        lst = [v for v in lst if v != 0]
    if lo is not None:
        lst = [v for v in lst if lo <= v <= hi]
    return lst


def _remove_outliers_te(x):
    """te.py DataLoad.remove_outliers 的忠实移植(滑动窗口方差>3连判3次删点)"""
    x = list(x)
    xi = [x[i:i + 3] for i in range(0, len(x), 1)]
    x_var = [float(np.var(w)) for w in xi]
    to_remove = []
    for j in range(0, len(x_var) - 2):
        if x_var[j] > 3 and x_var[j + 1] > 3 and x_var[j + 2] > 3:
            to_remove.append(x[j + 2])
    for v in to_remove:
        try:
            x.remove(v)
        except ValueError:
            pass
    return x


def _find_minimum_te(x, line_, prominence, width):
    """te.py find_minimum 的忠实移植，返回 (波谷索引, 是否存在)"""
    x = np.asarray(x, dtype=float)
    peaks, _ = find_peaks(-x, prominence=prominence, width=width)
    out = peaks
    peaks = peaks[x[peaks] <= line_]
    concave = []
    for p in peaks:
        if abs(x[p + 1] - x[p]) < CFG['LOCAL_THRESHOLD'] or abs(x[p - 1] - x[p]) < CFG['LOCAL_THRESHOLD']:
            concave.append(p)
    return np.array(concave), len(concave) > 0, out


def compute_flags_original(row):
    """te.py check_narrow_condition + compute_flags 的忠实移植，返回6个0/1"""
    flags = {k: 0 for k in FLAG_COLS}
    try:
        FM_aim_col, FM_aim_hot = float(row['FMWTARGETCOL']), float(row['FMWTARGETHOT'])
        FM_act_hot, PDI = float(row['FMWIDTHACTHOT']), float(row['PDIWIDTHTOL'])
        RM_aim_hot, RM_act = float(row['RDWTARGETTOTAL']), float(row['RMWIDTHACTHOT'])
        if any(np.isnan(v) for v in [FM_aim_col, FM_aim_hot, FM_act_hot, PDI, RM_aim_hot, RM_act]):
            return flags
    except (TypeError, KeyError):
        return flags

    fm = _seq_te(row['FMDATASEQ'], FM_aim_hot - 30, FM_aim_hot + 30, trim_first=True)
    rm = _seq_te(row['RMDATASEQ'], RM_aim_hot - 30, RM_aim_hot + 30)
    dc = _seq_te(row['DCDATASEQ'], FM_aim_hot - 30, FM_aim_hot + 30)
    dc = _remove_outliers_te(dc)
    if len(fm) < 5 or len(rm) < 5 or len(dc) < 5:
        return flags

    FM_mean = FM_act_hot / (FM_aim_hot / FM_aim_col)   # te.py 的 "FM_mean" = 冷值折算
    DC_mean = float(np.mean(dc))
    RM_mean = RM_act
    FMDELCOL = FM_aim_col - PDI + CFG['FIX']

    # te.py 主流程门控: FM/DC 全部点都高于下限 -> 直接全0
    if np.all(np.array(dc) > FMDELCOL) and np.all(np.array(fm) > FMDELCOL):
        return flags

    p_FM, is_FM, FM_min_all = _find_minimum_te(fm, FMDELCOL, CFG['PROMINENCE'], CFG['WIDTH'])
    p_RM, is_RM, _ = _find_minimum_te(rm, RM_mean, CFG['PROMINENCE_RM'], CFG['WIDTH_RM'])
    p_DM, is_DM, _ = _find_minimum_te(dc, FMDELCOL, CFG['PROMINENCE'], CFG['WIDTH'])
    flags['is_FM'], flags['is_RM'], flags['is_DM'] = int(is_FM), int(is_RM), int(is_DM)

    # 整体窄: 均值贴近下限, 覆盖其他标志
    if abs(FM_mean - FMDELCOL) < CFG['NARROW_THRESHOLD'] or abs(DC_mean - FMDELCOL) < CFG['NARROW_THRESHOLD']:
        flags['is_NA'] = 1
        return flags

    if is_FM and not is_RM:
        return flags                                   # 精轧拉钢形态
    if is_FM and is_RM:
        rot_F0, rot_R0 = abs(p_FM[0] / len(fm)), abs(p_RM[0] / len(rm))
        rot_F1, rot_R1 = abs(p_FM[-1] / len(fm)), abs(p_RM[-1] / len(rm))
        if (rot_F0 < CFG['HEADROTATION'] and rot_R0 < CFG['HEADROTATION']) or \
           (rot_F1 > CFG['TAILROTATION'] and rot_R1 > CFG['TAILROTATION']):
            flags['is_HT'] = 1                         # 头尾同位置波谷
            return flags
        rot = int(len(rm) * 0.1)
        mid = np.array(rm[rot:-rot])
        order = np.argsort(mid)
        if abs(mid[order[5]] - mid[order[-5]]) > CFG['DH']:
            flags['is_WS'] = 1                         # 中部不均匀
        return flags
    if is_DM and not is_FM:
        fm_arr = np.array(fm)
        peaks = FM_min_all[fm_arr[FM_min_all] <= FMDELCOL + CFG['AIM_COL_PLUS']]
        if len(peaks) > 0:
            return flags
        if dc[p_DM[0]] < FMDELCOL:                     # LOCAL_RIGHT=1, 即单点判断
            return flags
        if abs(np.mean(fm) - np.mean(dc)) > CFG['SWELL_THRESHOLD']:
            return flags
    return flags


# ---------------------------------------------------------------- 主入口

def extract_all(width_csv):
    """读取 width.csv，输出 (特征DataFrame, 标签Series, STRIPNO)"""
    df = pd.read_csv(width_csv, encoding='utf-8')
    rows, labels, stripnos = [], [], []
    for _, row in df.iterrows():
        f = {c: row[c] for c in BASE_COLS}
        f.update(compute_flags_original(row))
        curves = {'RM': clean_curve(parse_curve(row['RMDATASEQ'])),
                  'FM': clean_curve(parse_curve(row['FMDATASEQ'])),
                  'DC': clean_curve(parse_curve(row['DCDATASEQ']))}
        targets = {'RM': row['RDWTARGETTOTAL'], 'FM': row['FMWTARGETHOT'], 'DC': row['FMWTARGETHOT']}
        for stage in ('RM', 'FM', 'DC'):
            f.update(stage_features(row[f'{stage}DATASEQ'], targets[stage], stage))
        f.update(cross_features(row, curves))
        rows.append(f)
        labels.append(row['label'])
        stripnos.append(row['STRIPNO'])

    X = pd.DataFrame(rows)
    for c in ALL_FEATS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[ALL_FEATS].astype(float)
    X.insert(0, 'STRIPNO', stripnos)
    X['label'] = labels
    return X


if __name__ == '__main__':
    import os, time
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    feats = extract_all(os.path.join(base, 'data', 'width.csv'))
    print(f'提取完成: {feats.shape}, 耗时 {time.time()-t0:.1f}s')
    out = os.path.join(base, 'results', 'features.csv')
    feats.to_csv(out, index=False, encoding='utf-8-sig')
    print('已保存:', out)
    print(feats[['STRIPNO', 'label'] + FLAG_COLS].head(8).to_string())
