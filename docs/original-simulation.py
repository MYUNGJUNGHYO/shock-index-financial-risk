import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1) 데이터 생성: "급등 + 급락 혼합(Mixed Jumps)" 시뮬레이션
# ============================================================
def simulate_price_mixed_jumps(
    T=1200,                # 일수
    S0=100.0,              # 초기 주가
    mu=0.0002,             # 드리프트(일별)
    sigma=0.01,            # 정상 변동성(일별)
    p_jump=0.02,           # 점프 발생 확률(일별)
    jump_mag=0.06,         # 점프 크기
    jump_sigma=0.04,       # 점프 표준편차
    seed=7
):
    rng = np.random.default_rng(seed)
    S = np.zeros(T+1)
    r = np.zeros(T)
    true_jump = np.zeros(T, dtype=int)
    jump_size = np.zeros(T)

    S[0] = S0

    for t in range(T):
        diff = rng.normal(mu, sigma)
        j = 1 if rng.random() < p_jump else 0
        true_jump[t] = j

        if j == 1:
            # 50% 확률로 급등(+) 또는 급락(-) 결정!
            direction = 1 if rng.random() < 0.5 else -1
            J = direction * np.abs(rng.normal(jump_mag, jump_sigma))
        else:
            J = 0.0

        jump_size[t] = J
        r[t] = diff + J
        S[t+1] = S[t] * np.exp(r[t])

    df = pd.DataFrame({
        "S": S[1:], "r": r,
        "true_jump": true_jump,
        "jump_size": jump_size
    })
    df.index.name = "t"
    return df

# ============================================================
# 2) Shock 지표 q_t 계산 (학부 수준: rolling std 사용)
#    q_t = |r_t| / sigma_hat_{t-1}
# ============================================================

def compute_shock_index(df, vol_window=20):
    r = df["r"]
    sigma_hat = r.rolling(vol_window).std()
    q = (r.abs() / sigma_hat.shift(1))
    df = df.copy()
    df["sigma_hat"] = sigma_hat
    df["q"] = q
    return df

# ============================================================
# 3) SDI-SVJ: 점프 강도(확률) 모델
#    lambda_t = lambda0 * exp(gamma * q_t)
#
#    여기서는 "확률"처럼 쓰기 위해
#    prob_jump_t = 1 - exp(-lambda_t) (dt=1 가정)
# ============================================================

def sdi_intensity_and_prob(df, lambda0=0.02, gamma=0.8):
    df = df.copy()
    q = df["q"].fillna(0.0)
    lam = lambda0 * np.exp(gamma * q)
    prob = 1.0 - np.exp(-lam)  # dt=1 근사
    df["lambda_t"] = lam
    df["p_jump_hat"] = prob
    return df


# ============================================================
# 4) 성능 평가(정확성): threshold 바꿔가며 Precision/Recall, ROC 계산
# ============================================================

def compute_pr_roc(y_true, score, n_grid=200):
    """
    y_true: 0/1
    score:  continuous [0,1]  (predicted probability)
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    # threshold grid (높을수록 점프라고 예측)
    ths = np.linspace(0.0, 1.0, n_grid)

    precisions, recalls = [], []
    tprs, fprs = [], []

    P = y_true.sum()
    N = len(y_true) - P

    for th in ths:
        y_pred = (score >= th).astype(int)

        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        tpr = TP / P if P > 0 else 0.0
        fpr = FP / N if N > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        tprs.append(tpr)
        fprs.append(fpr)

    return {
        "thresholds": ths,
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "tpr": np.array(tprs),
        "fpr": np.array(fprs),
    }


def auc_trapz(x, y):
    # simple AUC via trapezoid
    order = np.argsort(x)
    x2 = x[order]
    y2 = y[order]
    return np.trapz(y2, x2)


# ============================================================
# 5) 점프 영향 분석: (a) 분포 비교 (b) 꼬리리스크(VaR) 비교
# ============================================================

def var_es(returns, alpha=0.01):
    """
    VaR/ES for left tail (loss tail).
    returns: array-like
    alpha: e.g. 0.01 => 1% VaR
    """
    r = np.asarray(returns)
    var = np.quantile(r, alpha)
    es = r[r <= var].mean() if np.any(r <= var) else var
    return var, es


# ============================================================
# 6) 실행 및 이미지 생성 (7개 그래프 전체 포함)
# ============================================================
def main():
    # ---------- (A) 1) 데이터 생성 ----------
    df = simulate_price_mixed_jumps(
        T=1200, S0=100.0,
        mu=0.0002, sigma=0.01,
        p_jump=0.02, jump_mag=0.06, jump_sigma=0.04,
        seed=7
    )

    # --- (B) Shock index ---
    df = compute_shock_index(df, vol_window=20)

    # --- (C) SDI-SVJ 점프 확률 ---
    df = sdi_intensity_and_prob(df, lambda0=0.02, gamma=0.8)

    # --- (D) 베이스라인 ---
    p_base = df["true_jump"].mean()
    df["p_jump_base"] = p_base

    # --- (E) 정확성 평가 ---
    y_true = df["true_jump"].values
    score_sdi = df["p_jump_hat"].values
    score_base = df["p_jump_base"].values

    prroc_sdi = compute_pr_roc(y_true, score_sdi)
    prroc_base = compute_pr_roc(y_true, score_base)

    auc_roc_sdi = auc_trapz(prroc_sdi["fpr"], prroc_sdi["tpr"])
    auc_pr_sdi = auc_trapz(prroc_sdi["recall"], prroc_sdi["precision"])

    auc_roc_base = auc_trapz(prroc_base["fpr"], prroc_base["tpr"])
    auc_pr_base = auc_trapz(prroc_base["recall"], prroc_base["precision"])

    print("========== Accuracy Summary ==========")
    print(f"True jump rate (mean)     : {p_base:.4f}")
    print(f"ROC AUC  (SDI)            : {auc_roc_sdi:.4f}")
    print(f"ROC AUC  (Baseline const) : {auc_roc_base:.4f}")
    print(f"PR  AUC  (SDI)            : {auc_pr_sdi:.4f}")
    print(f"PR  AUC  (Baseline const) : {auc_pr_base:.4f}")
    print("======================================")

    # --- (F) 점프 영향 분석 데이터 ---
    r_all = df["r"].values
    r_nojump = df.loc[df["true_jump"] == 0, "r"].values
    r_jump = df.loc[df["true_jump"] == 1, "r"].values

    var1_all, es1_all = var_es(r_all)
    var1_no, es1_no = var_es(r_nojump)

    print("\n========== Jump Impact (Tail Risk) ==========")
    print(f"1% VaR (ALL)    : {var1_all:.4f} , ES: {es1_all:.4f}")
    print(f"1% VaR (NOJUMP) : {var1_no:.4f} , ES: {es1_no:.4f}")
    print("=============================================")

    # ---------- (G) 이미지 생성 ----------
    t = df.index.values
    up_idx = df[(df["true_jump"] == 1) & (df["jump_size"] > 0)].index
    down_idx = df[(df["true_jump"] == 1) & (df["jump_size"] < 0)].index

    # (1) 주가 그래프 (Up:^, Down:v)
    plt.figure(figsize=(11, 4))
    plt.plot(t, df["S"], linewidth=1, color='black', alpha=0.4)
    plt.scatter(up_idx, df.loc[up_idx, "S"], marker="^", color='red', s=40, label="Upside Jump")
    plt.scatter(down_idx, df.loc[down_idx, "S"], marker="v", color='blue', s=40, label="Downside Jump")
    plt.title("(1) Price with Mixed Jumps"); plt.legend(); plt.show()

    # (2) Shock Index q_t
    plt.figure(figsize=(11, 4))
    plt.plot(t, df["q"].values, linewidth=1)
    plt.scatter(df.index[df["true_jump"]==1], df.loc[df["true_jump"]==1, "q"], marker="x", color='orange', s=30)
    plt.title("(2) Shock Index q_t (x = TRUE jumps)"); plt.show()

    # (3) Predicted Jump Probability
    plt.figure(figsize=(11, 4))
    plt.plot(t, df["p_jump_hat"], label="SDI-SVJ p_hat")
    plt.plot(t, df["p_jump_base"], label="Baseline")
    plt.scatter(df.index[df["true_jump"]==1], df.loc[df["true_jump"]==1, "p_jump_hat"], marker="x", color='red')
    plt.title("(3) Predicted Jump Probability vs Baseline"); plt.legend(); plt.show()

    # (4) ROC Curve
    plt.figure(figsize=(6, 6))
    plt.plot(prroc_sdi["fpr"], prroc_sdi["tpr"], label=f"SDI (AUC={auc_roc_sdi:.3f})")
    plt.plot([0,1],[0,1], linestyle="--"); plt.title("(4) ROC Curve"); plt.legend(); plt.show()

    # (5) Precision-Recall Curve
    plt.figure(figsize=(6, 6))
    plt.plot(prroc_sdi["recall"], prroc_sdi["precision"], label=f"SDI (AUC~{auc_pr_sdi:.3f})")
    plt.title("(5) Precision-Recall Curve"); plt.legend(); plt.show()

    # (6) Return Distribution
    plt.figure(figsize=(10, 4))
    plt.hist(r_nojump, bins=50, alpha=0.7, label="No-jump returns")
    plt.hist(r_jump, bins=50, alpha=0.7, label="Jump returns (Mixed)")
    plt.title("(6) Return Distribution"); plt.legend(); plt.show()

    # (7) 꼬리 위험 (VaR)
    plt.figure(figsize=(10, 4))
    plt.hist(r_all, bins=60, alpha=0.8)
    plt.axvline(var1_all, color='red', linestyle="--", label=f"VaR 1% (ALL)={var1_all:.3f}")
    plt.axvline(var1_no, color='blue', linestyle="--", label=f"VaR 1% (NOJUMP)={var1_no:.3f}")
    plt.title("(7) Tail Risk Impact (1% VaR)"); plt.legend(); plt.show()

if __name__ == "__main__":
    main()