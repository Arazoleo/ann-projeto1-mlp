"""Fecha a regressão: degradação por janela, gráficos e a única passada no teste."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from setup import preparar, metricas_regressao, SEEDS_TREINO
from graficos import treinar_uma_vez, plotar_ajuste
from baseline import plotar_curvas

EPOCAS = 10000
CONFIGS = {
    "baseline":     {"larguras": [128, 128], "lr": 0.05},
    "L1 1e-6":      {"larguras": [128, 128], "lr": 0.05, "l1": 1e-6},
    "L2 1e-5":      {"larguras": [128, 128], "lr": 0.05, "l2": 1e-5},
    "dropout 0.1":  {"larguras": [128, 128], "lr": 0.05, "dropout": 0.1},
    "momentum 0.5": {"larguras": [128, 128], "lr": 0.05, "momentum": 0.5},
}


def degradacao(hist, janela=1000):
    """Compara MÉDIAS de janelas, não o mínimo pontual (que o ruído enviesa)."""
    v = np.array(hist["val"])
    meio = v[len(v) // 2: len(v) // 2 + janela].mean()
    fim = v[-janela:].mean()
    return meio, fim, fim / meio


def agregar(lista, chave):
    vals = [d[chave] for d in lista]
    return float(np.mean(vals)), float(np.std(vals))


if __name__ == "__main__":
    dados = preparar(n=300)
    saida = {}

    for nome, cfg in CONFIGS.items():
        val, teste, degr = [], [], []
        for seed in SEEDS_TREINO:
            modelo, hist = treinar_uma_vez(dados, cfg=cfg, seed=seed, epocas=EPOCAS)
            val.append(metricas_regressao(modelo, dados["val"], dados["py"]))
            teste.append(metricas_regressao(modelo, dados["teste"], dados["py"]))
            degr.append(degradacao(hist)[2])
            if nome == "baseline" and seed == SEEDS_TREINO[0]:
                plotar_curvas(hist, "Baseline [128,128] lr=0.05").savefig(
                    "curva_baseline.png", dpi=130, bbox_inches="tight")
                plotar_ajuste(modelo, dados).savefig(
                    "ajuste_baseline.png", dpi=130, bbox_inches="tight")
                plotar_ajuste(modelo, dados, zoom=(1.0, 3.0)).savefig(
                    "ajuste_zoom_degrau.png", dpi=130, bbox_inches="tight")

        r2v, r2v_s = agregar(val, "R2")
        r2t, r2t_s = agregar(teste, "R2")
        mae, mae_s = agregar(teste, "MAE")
        rmse, rmse_s = agregar(teste, "RMSE")
        mse, mse_s = agregar(teste, "MSE")
        print(f"{nome:14s} val R2={r2v:.4f}±{r2v_s:.4f} | TESTE R2={r2t:.4f}±{r2t_s:.4f} "
              f"MAE={mae:.4f} RMSE={rmse:.4f} MSE={mse:.4f} | degrad={np.mean(degr):.3f}")
        saida[nome] = {"config": cfg,
                       "val_R2": [r2v, r2v_s], "teste_R2": [r2t, r2t_s],
                       "teste_MAE": [mae, mae_s], "teste_RMSE": [rmse, rmse_s],
                       "teste_MSE": [mse, mse_s],
                       "degradacao": float(np.mean(degr))}

    json.dump(saida, open("final.json", "w"), indent=2, default=str)
