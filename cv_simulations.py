#!/usr/bin/env python3
"""
Simulations and figures for the paper
  "Photonic Loop Memory with Logged-Displacement Rollback and Digital-Twin
   Calibration for Continuous-Variable Quantum Communication"

Run:   python cv_final_simulations.py            (figures go to ./figures)
       python cv_final_simulations.py --out ../Images
Needs: numpy, matplotlib   (pip install numpy matplotlib)

Conventions
-----------
* Coherent state |alpha>, alpha = (<X> + i<P>)/2, [X,P] = 2i, vacuum variance of X is 1.
* Fidelity of two coherent states: F = |<alpha|beta>|^2 = exp(-|alpha-beta|^2).
* Process noise eps_k is circular complex Gaussian with E|eps_k|^2 = sigma^2.
  The mean fidelity of such an error is E[exp(-|d|^2)] = 1/(1+v), v = E|d|^2.
* Loop: alpha_{n+1} = lam*alpha_n + dalpha_n + eps_n,  lam = sqrt(eta*G).
* Quantum-limited amplified loop (G = 1/eta): E|eps|^2 = (1-eta)/eta per lap.
* All Monte Carlo curves average R = 2000 realizations, seed 2026.
"""
import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- constants
SEED = 2026
R = 2000                    # Monte Carlo realizations
V_G = 2.0e8                 # group velocity in fiber (m/s)
M_MODES = 8                 # qumodes (pulses) in the loop
TAU = 1e-9                  # pulse spacing (s)
T_LOOP = M_MODES * TAU      # one lap = 8 ns
ELL = V_G * T_LOOP          # loop length = 1.6 m
FIBER_DB_KM = 0.2           # fiber attenuation (dB/km)
SWITCH_DB = 0.1             # assumed insertion loss per switch pass (dB)

rng = np.random.default_rng(SEED)
OUT = "figures"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.8,
})
COL = (3.5, 2.3)            # single-column figure (inches)
WIDE = (7.2, 2.7)           # double-column figure (inches)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300)
    plt.close(fig)


def cgauss(shape, var):
    """circular complex Gaussian with E|z|^2 = var"""
    s = math.sqrt(var / 2.0)
    return s * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


# ------------------------------------------------ Fig. 2: forward fidelity
def fig_forward():
    N = 1000
    n = np.arange(0, N + 1)
    fig, ax = plt.subplots(figsize=COL)
    for eta, c in ((0.9999, "tab:blue"), (0.999, "tab:orange"), (0.99, "tab:green")):
        s2 = (1 - eta) / eta                       # quantum-limited E|eps|^2 per lap
        d = np.cumsum(cgauss((R, N), s2), axis=1)  # accumulated shift
        d = np.concatenate([np.zeros((R, 1)), d], axis=1)
        F = np.exp(-np.abs(d) ** 2).mean(axis=0)   # Monte Carlo mean fidelity
        ax.plot(n, F, color=c, label=rf"$1-\eta=10^{{{int(round(math.log10(1 - eta)))}}}$")
        ax.plot(n, 1 / (1 + n * s2), "k--", lw=0.9)
    ax.plot([], [], "k--", lw=0.9, label=r"$1/(1+n\sigma_Q^2)$")
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Number of laps $n$")
    ax.set_ylabel(r"Retrieval fidelity $F_n$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(fig, "Figure_fwd_quantum_limited.png")


# ------------------------------------------------ Fig. 7: rollback (Monte Carlo)
def rollback_var(D, lam, se, st):
    k = np.arange(1, D + 1)
    return st ** 2 + se ** 2 * np.sum(lam ** (-2.0 * k))


def fig_rollback(lam=0.98, se=0.008, st=0.008, D=50):
    k = np.arange(1, D + 1)
    eps = cgauss((R, D), se ** 2)
    est = cgauss((R, 1), st ** 2)
    delta = est - np.cumsum(eps * lam ** (-k), axis=1)   # twin error minus accumulated noise
    F = np.exp(-np.abs(delta) ** 2).mean(axis=0)
    var = st ** 2 + se ** 2 * np.cumsum(lam ** (-2.0 * k))
    fig, ax = plt.subplots(figsize=COL)
    ax.plot(k, F, "-o", color="navy", ms=2.5)
    ax.set_xlim(0.5, D + 0.5)
    lo = math.floor((F.min() - 0.0005) * 1000) / 1000
    ax.set_ylim(lo, 1.0005)
    ax.set_xlabel("Rollback depth $D$")
    ax.set_ylabel(r"Rollback fidelity $F_{\mathrm{RB}}$")
    ax.grid(True, alpha=0.3)
    save(fig, "Figure_rollback_zoom.png")
    print("rollback: F(D=50) MC = %.4f, exp(-Var) = %.4f, 1/(1+Var) = %.4f, max|MC-exp(-Var)| = %.1e"
          % (F[-1], math.exp(-var[-1]), 1 / (1 + var[-1]), np.max(np.abs(F - np.exp(-var)))))


# ------------------------------------------------ Fig. 8: rollback (analytic)
def fig_analytic():
    fig, axs = plt.subplots(1, 2, figsize=WIDE)
    ax = axs[0]
    Ds = np.arange(1, 151)
    for lam, ls in zip((1.0, 0.995, 0.99, 0.98), ("-", "--", "-.", ":")):
        ax.plot(Ds, [math.exp(-rollback_var(d, lam, .008, .008)) for d in Ds], ls,
                label=rf"$\lambda={lam}$")
    ax.set_xlabel("Rollback depth $D$")
    ax.set_ylabel(r"Rollback fidelity $F_{\mathrm{RB}}$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_xlim(1, 150)
    ax = axs[1]
    lams = np.linspace(0.95, 1.0, 101)
    for se, ls in zip((0.004, 0.008, 0.016), ("-", "--", ":")):
        ax.plot(lams, [math.exp(-rollback_var(50, l, se, .008)) for l in lams], ls,
                label=rf"$\sigma_\epsilon={se}$")
    ax.set_xlabel(r"Propagation factor $\lambda$")
    ax.set_ylabel(r"$F_{\mathrm{RB}}$ at $D=50$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(0.95, 1.0)
    save(fig, "Figure_analytic_stability.png")
    lim = -math.log(0.99)
    for lam in (1.0, 0.995, 0.99, 0.98):
        d = 0
        while rollback_var(d + 1, lam, .008, .008) <= lim and d < 5000:
            d += 1
        print("rollback horizon lambda=%.3f  D_max(F>=0.99)=%d" % (lam, d))


# ------------------------------------------------ lifetime formulas
def n_life_amp(Fth, loss):
    """laps until the amplified loop reaches fidelity Fth"""
    eta = 1 - loss
    return (1 / Fth - 1) * eta / loss


def n_life_noamp(Fth, loss, alpha):
    """laps until an unamplified loop (pure shrinkage) reaches fidelity Fth"""
    eta = 1 - loss
    x = 1 - math.sqrt(-math.log(Fth)) / alpha
    return 2 * math.log(x) / math.log(eta) if x > 0 else float("inf")


# ------------------------------------------------ Fig. 3: link budget
def fig_comm():
    loss = 1e-3
    eta = 1 - loss
    n = np.arange(0, 2501)
    t_us = n * T_LOOP * 1e6
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axs[0]
    ax.plot(t_us, 1 / (1 + n * loss / eta), "k-", lw=2, label="amplified (quantum-limited)")
    for a, ls in ((0.5, "--"), (1.0, "-."), (3.0, ":")):
        ax.plot(t_us, np.exp(-a ** 2 * (1 - eta ** (n / 2)) ** 2), ls,
                label=rf"no amplifier, $|\alpha|={a}$")
    ax.set_xlim(0, t_us[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Storage time ($\mu$s)")
    ax.set_ylabel("Retrieval fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    top = ax.secondary_xaxis("top", functions=(lambda t: t * V_G * 1e-6 / 1e3,
                                               lambda L: L * 1e3 / (V_G * 1e-6)))
    top.set_xlabel("Equivalent fiber length (km)")
    ax = axs[1]
    ls_ = np.logspace(-5, -1, 200)
    for Fth, ls in ((0.9, "-"), (0.95, "--"), (0.99, ":")):
        ax.loglog(ls_, [n_life_amp(Fth, l) * T_LOOP * 1e6 for l in ls_], ls,
                  label=rf"$F_{{\mathrm{{th}}}}={Fth}$")
    ax.set_xlabel(r"Round-trip loss $1-\eta$")
    ax.set_ylabel(r"Maximum storage time ($\mu$s)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, "Figure_comm_budget.png")

    print("loop length = %.2f m, 1 us of storage = %.0f m of fiber" % (ELL, V_G * 1e-6))
    print("fiber-only loss per lap at 0.2 dB/km: %.2e" % (1 - 10 ** (-FIBER_DB_KM * ELL / 1e3 / 10)))
    for Fth in (0.9, 0.95):
        for l in (1e-2, 1e-3, 1e-4):
            n_ = n_life_amp(Fth, l)
            print("Fth=%.2f 1-eta=%.0e amplified: n=%.0f  t=%.2f us  L=%.1f m" %
                  (Fth, l, n_, n_ * T_LOOP * 1e6, n_ * ELL))
    for a in (0.5, 1, 3, 6):
        print("no-amp Fth=0.9 |a|=%.1f: n(1e-2)=%.0f ; n(1e-3)=%.0f" %
              (a, n_life_noamp(0.9, 1e-2, a), n_life_noamp(0.9, 1e-3, a)))
    for Fth in (0.9, 0.95, 0.99):
        print("alpha_c(Fth=%.2f)=%.2f" % (Fth, 2 * math.sqrt(-math.log(Fth)) / (1 / Fth - 1)))
    a_fib = FIBER_DB_KM / 4.343 / 1e3
    print("fiber-only L_max (F_th=0.9) = %.2f km" % ((1 / 0.9 - 1) / a_fib / 1e3))


# ------------------------------------------------ Fig. 4: feed-forward latency
def fig_ff():
    t = np.linspace(0, 2.0, 400)                 # latency in microseconds
    n = t * 1e-6 / T_LOOP
    fig, ax = plt.subplots(figsize=COL)
    for loss, ls in ((1e-2, ":"), (1e-3, "-"), (1e-4, "--")):
        eta = 1 - loss
        ax.plot(t, 1 / (1 + n * loss / eta), ls,
                label=rf"$1-\eta=10^{{{int(round(math.log10(loss)))}}}$")
    ax.axhline(0.9, color="gray", lw=0.8)
    ax.set_xlim(0, 2)
    ax.set_ylim(0.4, 1.005)
    ax.set_xlabel(r"Classical-result latency ($\mu$s)")
    ax.set_ylabel("Fidelity at feed-forward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    save(fig, "Figure_feedforward.png")
    for tp in (0.1, 0.2, 0.5, 1.0):
        nn = tp * 1e-6 / T_LOOP
        print("latency %.1f us: n=%.1f ; F(1e-2)=%.3f F(1e-3)=%.3f F(1e-4)=%.4f" % (
            tp, nn, 1 / (1 + nn * 1e-2 / 0.99), 1 / (1 + nn * 1e-3 / 0.999), 1 / (1 + nn * 1e-4 / 0.9999)))


# ------------------------------------------------ Fig. 5: loop vs fiber delay lines
def delay_comparison():
    """Mean total power-loss fraction for feed-forward latency uniform in [0.1, 0.5] us."""
    tmin, tmax = 0.1e-6, 0.5e-6
    sf = 1 - 10 ** (-SWITCH_DB / 10)                 # loss of one switch pass
    m = 6                                            # binary stages: 2^6 = 64 >= 62.5 steps
    tt = np.linspace(tmin, tmax, 20001)
    laps = np.ceil(tt / T_LOOP)
    nbar = laps.mean()                               # mean laps (about 38)
    fib = lambda t: 1 - 10 ** (-FIBER_DB_KM * (V_G * t / 1e3) / 10)   # fiber loss for delay t
    spool = fib(tmax)
    bank = 1 - (1 - sf) ** (2 * m) * (1 - np.mean(fib(laps * T_LOOP)))
    loop = lambda eps: 1 - (1 - sf) ** 2 * (1 - eps) ** nbar
    return dict(sf=sf, m=m, nbar=nbar, spool=spool, bank=bank, loop=loop)


def fig_delaycomp():
    d = delay_comparison()
    eps = np.logspace(-5, -1, 200)
    fig, ax = plt.subplots(figsize=COL)
    ax.loglog(eps, d["loop"](eps), "-", color="tab:blue", label="loop (this work)")
    ax.axhline(d["bank"], ls="--", color="tab:orange", label="switched delay bank")
    ax.axhline(d["spool"], ls=":", lw=2, color="tab:green", label="fixed fiber spool")
    ax.set_xlabel(r"Loss per lap $\epsilon$")
    ax.set_ylabel("Mean total loss fraction")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    save(fig, "Figure_delay_comparison.png")
    rhs = (1 - d["bank"]) / (1 - d["sf"]) ** 2
    eps_b = 1 - rhs ** (1 / d["nbar"])
    print("DELAYCOMP nbar=%.2f switch loss=%.4f spool=%.4f bank=%.4f loop(1e-3)=%.4f loop(1e-4)=%.4f "
          "loop(0)=%.4f eps_break_vs_bank=%.2e" % (d["nbar"], d["sf"], d["spool"], d["bank"],
                                                  d["loop"](1e-3), d["loop"](1e-4), d["loop"](0.0), eps_b))
    print("fiber used: spool %.0f m, bank %.1f m, loop %.1f m" % (V_G * 0.5e-6, ELL * (2 ** d["m"] - 1), ELL))


# ------------------------------------------------ Fig. 9: entangled (squeezed) payload
def nu_minus(a, b, c):
    """smallest symplectic eigenvalue of the partially transposed covariance matrix
    sigma = [[a I, c Z], [c Z, b I]]  (vacuum = 1, Z = diag(1,-1))"""
    D = a * a + b * b + 2 * c * c
    det = (a * b - c * c) ** 2
    return np.sqrt((D - np.sqrt(np.maximum(D * D - 4 * det, 0.0))) / 2.0)


def log_neg(nu):
    return np.maximum(0.0, -np.log(nu))


def ent_amplified(r, eps, n):
    """one half of a two-mode squeezed vacuum (squeezing r) in the amplified loop (G = 1/eta, lam = 1)"""
    C, S = math.cosh(2 * r), math.sinh(2 * r)
    eta = 1 - eps
    return nu_minus(C, C + 2 * n * eps / eta, S)       # added noise 2*eps/eta per lap, correlation unchanged


def ent_passive(r, eps, n):
    """same state in a passive loop (no amplifier): pure loss, t = eta^n"""
    C, S = math.cosh(2 * r), math.sinh(2 * r)
    t = (1 - eps) ** n
    return nu_minus(C, t * C + (1 - t), np.sqrt(t) * S)


def tele_amplified(r, eps, n):
    """unit-gain CV teleportation fidelity of a coherent state, resource = squeezed pair whose half
    waited n laps in the amplified loop:  F = 1/(1 + (a + b - 2c)/2) = 1/(1 + exp(-2r) + n*eps/eta)"""
    return 1.0 / (1.0 + math.exp(-2 * r) + n * eps / (1 - eps))


def tele_passive(r, eps, n):
    C, S = math.cosh(2 * r), math.sinh(2 * r)
    t = (1 - eps) ** n
    return 1.0 / (1.0 + (C + t * C + (1 - t) - 2 * np.sqrt(t) * S) / 2.0)


def fig_entanglement():
    eps = 1e-3
    n = np.arange(0, 2001)
    t_us = n * T_LOOP * 1e6
    fig, axs = plt.subplots(1, 2, figsize=WIDE)
    ax = axs[0]
    for dB, c in ((3, "tab:blue"), (6, "tab:orange"), (10, "tab:green")):
        r = dB / 8.686                                   # squeezing in dB = 10*log10(exp(2r))
        ax.plot(t_us, log_neg(ent_amplified(r, eps, n)), "-", color=c, label=f"{dB} dB")
        ax.plot(t_us, log_neg(ent_passive(r, eps, n)), "--", color=c)
    ax.plot([], [], "k-", label="amplified loop")
    ax.plot([], [], "k--", label="passive loop")
    ax.set_xlim(0, t_us[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Storage time ($\mu$s)")
    ax.set_ylabel(r"Logarithmic negativity $E_N$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2)
    ax = axs[1]
    for dB, c in ((3, "tab:blue"), (6, "tab:orange"), (10, "tab:green")):
        r = dB / 8.686
        ax.plot(t_us, tele_amplified(r, eps, n), "-", color=c, label=f"{dB} dB")
        ax.plot(t_us, np.array([tele_passive(r, eps, k) for k in n]), "--", color=c)
    ax.axhline(0.5, color="gray", lw=0.9)
    ax.set_xlim(0, t_us[-1])
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel(r"Storage time ($\mu$s)")
    ax.set_ylabel("Teleportation fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, "Figure_entanglement.png")
    for dB in (3, 6, 10):
        r = dB / 8.686
        n_cl = (1 - math.exp(-2 * r)) * (1 - eps) / eps
        print("teleportation %2d dB: F(n=0)=%.3f ; amplified loop stays above 1/2 until n=%.0f laps (%.1f us); "
              "passive F(n=500)=%.3f amplified F(n=500)=%.3f" % (dB, tele_amplified(r, eps, 0), n_cl,
              n_cl * T_LOOP * 1e6, tele_passive(r, eps, 500), tele_amplified(r, eps, 500)))
    for dB in (3, 6, 10):
        r = dB / 8.686
        for e in (1e-3, 1e-4):
            nn = 0
            while ent_amplified(r, e, nn) < 1 and nn < 10 ** 7:
                nn += 1
            print("squeeze %2d dB eps=%.0e: amplified loop entanglement ends at n=%d laps (%.2f us); "
                  "closed form eta/eps = %.0f" % (dB, e, nn, nn * T_LOOP * 1e6, (1 - e) / e))
    for dB in (3, 6, 10):
        r = dB / 8.686
        print("E_N (eps=1e-3) %2d dB: n=0: %.3f | n=100: amp %.3f pas %.3f | n=500: amp %.3f pas %.3f | "
              "n=900: amp %.3f pas %.3f" % (dB, log_neg(ent_amplified(r, 1e-3, 0)),
              log_neg(ent_amplified(r, 1e-3, 100)), log_neg(ent_passive(r, 1e-3, 100)),
              log_neg(ent_amplified(r, 1e-3, 500)), log_neg(ent_passive(r, 1e-3, 500)),
              log_neg(ent_amplified(r, 1e-3, 900)), log_neg(ent_passive(r, 1e-3, 900))))


# ------------------------------------------------ amplitude trajectories (Fig. 6)
def fig_trajectories():
    """|alpha_i| of the M = 8 qumodes over 1000 frames, one realization.
    (a) ideal: lam = 1, no noise;  (b) lam = 0.997, sigma_eps = 0.02.
    Same displacement steps (sigma_D = 0.03 per frame) in both panels.
    Uses its own generator so the other figures are unchanged."""
    g = np.random.default_rng(6)
    N, sD, lam, se = 1000, 0.03, 0.997, 0.02
    a0 = g.uniform(1.0, 1.8, M_MODES) * np.exp(1j * g.uniform(0, 2*np.pi, M_MODES))
    D = (g.normal(0, sD/np.sqrt(2), (N, M_MODES)) + 1j*g.normal(0, sD/np.sqrt(2), (N, M_MODES)))
    E = (g.normal(0, se/np.sqrt(2), (N, M_MODES)) + 1j*g.normal(0, se/np.sqrt(2), (N, M_MODES)))
    ai, an = [a0], [a0]
    for n in range(N):
        ai.append(ai[-1] + D[n])
        an.append(lam*an[-1] + D[n] + E[n])
    ai, an = np.abs(np.array(ai)), np.abs(np.array(an))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for k in range(M_MODES):
        ax[0].plot(ai[:, k], lw=1.0, label="qumode %d" % (k + 1))
        ax[1].plot(an[:, k], lw=1.0)
    for a in ax:
        a.set_xlabel("frame number"); a.grid(alpha=.3)
    ax[0].set_ylabel(r"$|\alpha_i|$")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=8,
               frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "Figure_trajectories.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures", help="output folder for the PNG figures")
    OUT = ap.parse_args().out
    # NOTE: keep this order; the random numbers (seed 2026) are drawn in sequence.
    fig_forward()
    fig_rollback()
    fig_analytic()
    fig_comm()
    fig_ff()
    fig_delaycomp()
    fig_entanglement()
    fig_trajectories()
    print("done; figures written to", os.path.abspath(OUT))
