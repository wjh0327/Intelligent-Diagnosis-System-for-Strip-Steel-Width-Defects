# -*- coding: utf-8 -*-
"""
小型 1D-CNN 溯源模型
=================================
输入: 三段全长宽度曲线 (粗轧RM / 精轧FM / 卷取DC) —— 不经任何人工特征工程
预处理: 每段去0、掐头去尾各3点(与te.py清洗对齐) → 线性插值 resample 到 128 点
        → 按各自目标宽度归一为千分相对偏差 (x-T)/T*1000 → 截断到 ±50mm
模型: 3通道1D-CNN, 5层卷积(32/32/64/64/96) + 全局平均池化 + 两层全连接, 约5.4万参数
训练: 加权交叉熵(w(c)=N/(K·n(c)), 与LightGBM口径一致) + Adam + 早停
      早停指标与v3协议一致: 0.6*验证准确率 + 0.4*验证宏召回
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from features import parse_curve, clean_curve

RESAMPLE_N = 128
CLIP_MM = 40.0
JITTER_SIGMA = 0.8   # 训练时输入抖动(千分偏差), 轻微增强防过拟合
CLASS_ORDER = ['卷取拉钢', '整体宽', '整体窄', '楔形', '没有拉钢', '粗轧头尾', '精轧拉钢']
STAGE_TARGET = {'RM': 'RDWTARGETTOTAL', 'FM': 'FMWTARGETHOT', 'DC': 'FMWTARGETHOT'}
STAGE_CURVE = {'RM': 'RMDATASEQ', 'FM': 'FMDATASEQ', 'DC': 'DCDATASEQ'}


# ---------------------------------------------------------------- 数据管线

def resample_curve(x, n=RESAMPLE_N):
    """变长曲线线性插值到固定长度(头→尾等位置采样)"""
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) == 1:
        return np.full(n, x[0], dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, x).astype(np.float32)


def build_curve_tensor(df: pd.DataFrame) -> np.ndarray:
    """574条 → (N, 4, 128) float32。
    通道: RM偏差 / FM偏差 / DC偏差 / 位置编码(0→1, 让卷积核可感知头尾位置)"""
    out = np.zeros((len(df), 4, RESAMPLE_N), dtype=np.float32)
    out[:, 3, :] = np.linspace(0.0, 1.0, RESAMPLE_N, dtype=np.float32)[None, :]
    for i, (_, row) in enumerate(df.iterrows()):
        for c, stage in enumerate(('RM', 'FM', 'DC')):
            target = float(row[STAGE_TARGET[stage]])
            x = clean_curve(parse_curve(row[STAGE_CURVE[stage]]))
            if len(x) < 5 or target <= 0:
                continue                     # 数据缺失: 保持0(即贴目标), 与特征路线NaN口径对应
            x = np.clip(x, target - CLIP_MM, target + CLIP_MM)
            dev = (resample_curve(x) - target) / target * 1000.0   # 千分相对偏差
            out[i, c, :] = dev
    return out


# ---------------------------------------------------------------- 模型

class WidthCNN(nn.Module):
    """4通道(3曲线+1位置编码)1D-CNN, 约5.5万参数"""

    def __init__(self, n_classes=7, dropout=0.3, in_channels=4):
        super().__init__()
        def block(cin, cout, k):
            return [nn.Conv1d(cin, cout, k, padding=k // 2),
                    nn.BatchNorm1d(cout), nn.ReLU()]
        self.features = nn.Sequential(
            *block(in_channels, 32, 7), *block(32, 32, 5), nn.MaxPool1d(2),   # 128 -> 64
            *block(32, 64, 5), *block(64, 64, 3), nn.MaxPool1d(2),  # 64 -> 32
            *block(64, 96, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- 训练与评估

def class_weights(y, n_classes=7):
    """w(c) = N/(K·n(c)), 与v3的class_weight='balanced'同口径"""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.FloatTensor(w)


def mix_score(acc, macro_recall):
    return 0.6 * acc + 0.4 * macro_recall


def macro_recall_np(y_true, y_pred):
    rs = []
    for c in np.unique(np.concatenate([y_true, y_pred])):
        m = y_true == c
        if m.sum():
            rs.append(float((y_pred[m] == c).mean()))
    return float(np.mean(rs)) if rs else 0.0


def train_model(X_tr, y_tr, seed=42, max_epochs=200, patience=25,
                batch_size=32, lr=1e-3, weight_decay=1e-4, verbose=False):
    """在给定训练集上训练, 内部再切15%做早停验证; 返回最优state_dict"""
    from sklearn.model_selection import train_test_split
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.LongTensor(y_tr)
    idx_tr, idx_val = train_test_split(
        np.arange(len(y_tr)), test_size=0.15, random_state=seed, stratify=y_tr)

    w = class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=w)
    model = WidthCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds_tr = torch.utils.data.TensorDataset(X_tr_t[idx_tr], y_tr_t[idx_tr])
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g)

    best_score, best_state, bad = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            if JITTER_SIGMA > 0:      # 仅抖动3个曲线通道, 不动位置编码通道
                noise = torch.randn(xb.shape[0], 3, xb.shape[2]) * JITTER_SIGMA
                xb = xb.clone()
                xb[:, :3, :] = xb[:, :3, :] + noise
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        # 验证
        model.eval()
        with torch.no_grad():
            logits = model(X_tr_t[idx_val])
            pred = logits.argmax(1).numpy()
            yv = y_tr_t[idx_val].numpy()
        acc = float((pred == yv).mean())
        mrec = macro_recall_np(yv, pred)
        score = mix_score(acc, mrec)
        if score > best_score:
            best_score, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and (epoch + 1) % 20 == 0:
            print(f'    epoch {epoch+1}: val_acc={acc:.3f} val_mrec={mrec:.3f} score={score:.4f}')
    model.load_state_dict(best_state)
    return model


def predict_proba(model, X, batch_size=128):
    model.eval()
    outs = []
    X = torch.FloatTensor(X)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outs.append(torch.softmax(model(X[i:i + batch_size]), dim=1).numpy())
    return np.vstack(outs)


# ---------------------------------------------------------------- 混合模型: 曲线CNN + 12维特征

N_AUX = 12  # 6工艺参数 + 6规则flag


class HybridWidthCNN(nn.Module):
    """曲线CNN分支(GAP出96维) + 12维特征分支(线性出32维) 拼接后分类, 约6.3万参数"""

    def __init__(self, n_classes=7, dropout=0.3, in_channels=4, n_aux=N_AUX):
        super().__init__()
        def block(cin, cout, k):
            return [nn.Conv1d(cin, cout, k, padding=k // 2),
                    nn.BatchNorm1d(cout), nn.ReLU()]
        self.features = nn.Sequential(
            *block(in_channels, 32, 7), *block(32, 32, 5), nn.MaxPool1d(2),
            *block(32, 64, 5), *block(64, 64, 3), nn.MaxPool1d(2),
            *block(64, 96, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.aux = nn.Sequential(nn.Linear(n_aux, 32), nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.classifier = nn.Sequential(
            nn.Linear(96 + 32, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_curve, x_aux):
        h = self.features(x_curve).flatten(1)
        z = self.aux(x_aux)
        return self.classifier(torch.cat([h, z], dim=1))


def train_hybrid(X_curve, X_feat, y, seed=42, max_epochs=200, patience=25,
                 batch_size=32, lr=1e-3, weight_decay=1e-4):
    """训练混合模型。特征标准化仅在内部训练子集上拟合(防泄漏)。
    返回 (model, feat_mean, feat_std)"""
    from sklearn.model_selection import train_test_split
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    Xc = torch.FloatTensor(X_curve)
    Xf_all = np.asarray(X_feat, dtype=np.float32)
    y_t = torch.LongTensor(y)
    idx_tr, idx_val = train_test_split(
        np.arange(len(y)), test_size=0.15, random_state=seed, stratify=y)

    # 标准化: 只用训练子集统计量
    f_mean = Xf_all[idx_tr].mean(axis=0)
    f_std = Xf_all[idx_tr].std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    Xf = (Xf_all - f_mean) / f_std
    Xf = torch.FloatTensor(Xf)

    w = class_weights(y)
    criterion = nn.CrossEntropyLoss(weight=w)
    model = HybridWidthCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = torch.utils.data.TensorDataset(Xc[idx_tr], Xf[idx_tr], y_t[idx_tr])
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    best_score, best_state, bad = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, xf, yb in loader:
            if JITTER_SIGMA > 0:
                noise = torch.randn(xb.shape[0], 3, xb.shape[2]) * JITTER_SIGMA
                xb = xb.clone()
                xb[:, :3, :] = xb[:, :3, :] + noise
            optimizer.zero_grad()
            loss = criterion(model(xb, xf), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xc[idx_val], Xf[idx_val]).argmax(1).numpy()
        yv = y_t[idx_val].numpy()
        acc = float((pred == yv).mean())
        mrec = macro_recall_np(yv, pred)
        score = mix_score(acc, mrec)
        if score > best_score:
            best_score, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, f_mean, f_std


def predict_proba_hybrid(model, X_curve, X_feat, f_mean, f_std, batch_size=128):
    model.eval()
    Xf = (np.asarray(X_feat, dtype=np.float32) - f_mean) / f_std
    Xc = torch.FloatTensor(X_curve)
    Xf = torch.FloatTensor(Xf)
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xc), batch_size):
            outs.append(torch.softmax(model(Xc[i:i + batch_size], Xf[i:i + batch_size]), dim=1).numpy())
    return np.vstack(outs)


# ---------------------------------------------------------------- 仅特征MLP(消融用) + 梯度归因

class FeatureMLP(nn.Module):
    """仅12维特征分支的MLP(消融: 无曲线卷积分支), 约1.9千参数"""

    def __init__(self, n_classes=7, n_aux=N_AUX, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_aux, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, n_classes),
        )

    def forward(self, x_aux):
        return self.net(x_aux)


def train_mlp(X_feat, y, seed=42, max_epochs=200, patience=25, batch_size=32,
              lr=1e-3, weight_decay=1e-4):
    """训练仅特征MLP, 标准化仅在内部训练子集拟合。返回 (model, f_mean, f_std)"""
    from sklearn.model_selection import train_test_split
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    Xf_all = np.asarray(X_feat, dtype=np.float32)
    y_t = torch.LongTensor(y)
    idx_tr, idx_val = train_test_split(
        np.arange(len(y)), test_size=0.15, random_state=seed, stratify=y)

    f_mean = Xf_all[idx_tr].mean(axis=0)
    f_std = Xf_all[idx_tr].std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    Xf = torch.FloatTensor((Xf_all - f_mean) / f_std)

    w = class_weights(y)
    criterion = nn.CrossEntropyLoss(weight=w)
    model = FeatureMLP(n_aux=Xf.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = torch.utils.data.TensorDataset(Xf[idx_tr], y_t[idx_tr])
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    best_score, best_state, bad = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xf[idx_val]).argmax(1).numpy()
        yv = y_t[idx_val].numpy()
        score = mix_score(float((pred == yv).mean()), macro_recall_np(yv, pred))
        if score > best_score:
            best_score, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, f_mean, f_std


def predict_proba_mlp(model, X_feat, f_mean, f_std, batch_size=128):
    model.eval()
    Xf = torch.FloatTensor((np.asarray(X_feat, dtype=np.float32) - f_mean) / f_std)
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xf), batch_size):
            outs.append(torch.softmax(model(Xf[i:i + batch_size]), dim=1).numpy())
    return np.vstack(outs)


def gradient_attribution_hybrid(model, X_curve, X_feat, f_mean, f_std, device='cpu'):
    """梯度×输入归因(混合模型)。
    返回: aux_attr (N,12) 特征归因, curve_attr (N,3) 三段曲线通道归因(位置通道除外),
          pred (N,) 预测类别。归因 = 预测类logit对输入的梯度 × 输入值。"""
    model.eval()
    model.to(device)
    Xc = torch.FloatTensor(X_curve).to(device).requires_grad_(True)
    Xf = torch.FloatTensor((np.asarray(X_feat, dtype=np.float32) - f_mean) / f_std).to(device).requires_grad_(True)
    logits = model(Xc, Xf)
    pred = logits.argmax(1)
    sel = logits[torch.arange(len(pred)), pred].sum()
    sel.backward()
    aux_attr = (Xf.grad * Xf).detach().cpu().numpy()
    curve_attr = (Xc.grad * Xc).abs().detach().cpu().numpy()[:, :3, :].sum(axis=2)  # 位置通道除外; 先取绝对值避免正负贡献抵消
    return aux_attr, curve_attr, pred.cpu().numpy()
