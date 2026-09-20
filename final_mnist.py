"""Fecha a classificação: matriz de confusão, curvas e a única passada no teste."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from setup import criar_mlp, treinar, fixar_seed, DEVICE, SEEDS_TREINO
from mnist import preparar_mnist, metricas_classificacao
from baseline import plotar_curvas

EPOCAS = 60
CONFIGS = {
    "baseline":     {"larguras": [256, 256], "lr": 0.1},
    "L1 1e-6":      {"larguras": [256, 256], "lr": 0.1, "l1": 1e-6},
    "L2 1e-4":      {"larguras": [256, 256], "lr": 0.1, "l2": 1e-4},
    "dropout 0.2":  {"larguras": [256, 256], "lr": 0.1, "dropout": 0.2},
    "momentum 0.5": {"larguras": [256, 256], "lr": 0.1, "momentum": 0.5},
}


def treinar_uma_vez(dados, cfg, seed, epocas=EPOCAS):
    fixar_seed(seed)
    modelo = criar_mlp(784, 10, cfg["larguras"],
                       p_dropout=cfg.get("dropout", 0.0)).to(DEVICE)
    hist = treinar(modelo, dados["loader"], dados["val"], nn.CrossEntropyLoss(),
                   lr=cfg["lr"], epocas=epocas,
                   momentum=cfg.get("momentum", 0.0),
                   weight_decay=cfg.get("l2", 0.0),
                   l1_lambda=cfg.get("l1", 0.0))
    return modelo, hist


@torch.no_grad()
def plotar_confusao(modelo, dados, titulo, arquivo):
    modelo.eval()
    X, Y = dados["teste"][0].to(DEVICE), dados["teste"][1]
    pred = modelo(X).argmax(1).cpu().numpy()
    cm = confusion_matrix(Y.numpy(), pred, normalize="true")   # linhas somam 1
    classes = dados["classes"]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0.005:
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("previsto"); ax.set_ylabel("verdadeiro"); ax.set_title(titulo)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(arquivo, dpi=130, bbox_inches="tight")
    return cm


if __name__ == "__main__":
    dados = preparar_mnist()
    saida = {}

    for nome, cfg in CONFIGS.items():
        val, teste = [], []
        for seed in SEEDS_TREINO:
            modelo, hist = treinar_uma_vez(dados, cfg, seed)
            val.append(metricas_classificacao(modelo, dados["val"]))
            teste.append(metricas_classificacao(modelo, dados["teste"]))
            if seed == SEEDS_TREINO[0] and nome in ("baseline", "dropout 0.2"):
                plotar_curvas(hist, f"{nome} — Fashion-MNIST").savefig(
                    f"curva_mnist_{nome.replace(' ','_').replace('.','')}.png",
                    dpi=130, bbox_inches="tight")
                cm = plotar_confusao(
                    modelo, dados, f"Matriz de confusão (teste) — {nome}",
                    f"confusao_{nome.replace(' ','_').replace('.','')}.png")
                if nome == "baseline":
                    rec = {dados["classes"][i]: float(cm[i, i]) for i in range(10)}
                    piores = sorted(rec.items(), key=lambda kv: kv[1])[:3]
                    print("  recall por classe (3 piores):",
                          ", ".join(f"{c}={v:.3f}" for c, v in piores))
                    saida["recall_por_classe"] = rec

        ag = lambda lst, k: (float(np.mean([d[k] for d in lst])),
                             float(np.std([d[k] for d in lst])))
        av, avs = ag(val, "acuracia")
        at, ats = ag(teste, "acuracia")
        f1, f1s = ag(teste, "f1_macro")
        pr, _ = ag(teste, "precision_macro")
        rc, _ = ag(teste, "recall_macro")
        print(f"{nome:14s} val acc={av:.4f}±{avs:.4f} | TESTE acc={at:.4f}±{ats:.4f} "
              f"F1={f1:.4f} prec={pr:.4f} rec={rc:.4f}")
        saida[nome] = {"config": cfg, "val_acc": [av, avs], "teste_acc": [at, ats],
                       "teste_f1": [f1, f1s], "teste_precision": pr, "teste_recall": rc}

    json.dump(saida, open("final_mnist.json", "w"), indent=2, default=str)
