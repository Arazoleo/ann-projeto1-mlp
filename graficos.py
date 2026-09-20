import matplotlib
matplotlib.use("Agg")                   
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from setup import (preparar, criar_mlp, treinar, fixar_seed,
                   metricas_regressao, DEVICE)

EPOCAS = 10000
BASE = {"larguras": [128, 128], "lr": 0.05}


def treinar_uma_vez(dados, cfg=BASE, seed=1, epocas=EPOCAS):
    """rodar() devolve só métricas e descarta o modelo. Aqui guardamos o modelo."""
    fixar_seed(seed)
    modelo = criar_mlp(1, 1, cfg["larguras"],
                       p_dropout=cfg.get("dropout", 0.0)).to(DEVICE)
    hist = treinar(modelo, dados["loader"], dados["val"], nn.MSELoss(),
                   lr=cfg["lr"], epocas=epocas,
                   momentum=cfg.get("momentum", 0.0),
                   weight_decay=cfg.get("l2", 0.0),
                   l1_lambda=cfg.get("l1", 0.0))
    return modelo, hist


@torch.no_grad()
def plotar_ajuste(modelo, dados, zoom=None):
    modelo.eval()
    px, py = dados["px"], dados["py"]
    x_bruto, y_bruto = dados["bruto"]

    grade = np.linspace(-5, 5, 2000)
    f_real = np.sin(3 * grade) + 0.3 * grade + 1.5 * (grade > 2)   # sem ruído
    X = torch.tensor(px.transform(grade), dtype=torch.float32).view(-1, 1)
    pred = py.inverse(modelo(X.to(DEVICE)).cpu().numpy().ravel())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(x_bruto, y_bruto, s=12, alpha=0.35, c="gray",
               label="dados (com ruído)")
    ax.plot(grade, f_real, lw=2, label="f(x) verdadeira")
    ax.plot(grade, pred, lw=2, ls="--", label="predição da MLP")
    ax.axvline(2.0, c="red", ls=":", lw=1.2, label="descontinuidade (x=2)")
    if zoom:
        ax.set_xlim(*zoom)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8)
    return fig


if __name__ == "__main__":
    dados = preparar(n=300)
    modelo, hist = treinar_uma_vez(dados)

    print("validação:", metricas_regressao(modelo, dados["val"], dados["py"]))

    plotar_ajuste(modelo, dados).savefig(
        "ajuste_baseline.png", dpi=130, bbox_inches="tight")
    plotar_ajuste(modelo, dados, zoom=(1.0, 3.0)).savefig(
        "ajuste_zoom_degrau.png", dpi=130, bbox_inches="tight")
