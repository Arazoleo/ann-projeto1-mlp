import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
from setup import (criar_mlp, treinar, fixar_seed, DEVICE,
                   SEED_DADOS, SEEDS_TREINO)


def preparar_mnist(batch_size=128, n_val=10000):
    tr = datasets.FashionMNIST("./data", train=True,  download=True)
    te = datasets.FashionMNIST("./data", train=False, download=True)

    X   = tr.data.float().div(255.0).view(-1, 784)   # achatar 28x28 em 784
    Y   = tr.targets
    Xte = te.data.float().div(255.0).view(-1, 784)
    Yte = te.targets

    g = torch.Generator().manual_seed(SEED_DADOS)    # split
    idx = torch.randperm(len(X), generator=g)
    i_val, i_tr = idx[:n_val], idx[n_val:]
    Xtr, Ytr = X[i_tr], Y[i_tr]
    Xva, Yva = X[i_val], Y[i_val]

    mu, sd = Xtr.mean(), Xtr.std()                   # so do treino
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    loader = DataLoader(TensorDataset(Xtr, Ytr),
                        batch_size=batch_size, shuffle=True)
    return {"loader": loader, "treino": (Xtr, Ytr), "val": (Xva, Yva),
            "teste": (Xte, Yte), "classes": tr.classes,
            "mu": float(mu), "sd": float(sd)}


@torch.no_grad()
def metricas_classificacao(modelo, dados):
    modelo.eval()
    X, Y = dados[0].to(DEVICE), dados[1].to(DEVICE)
    pred = modelo(X).argmax(dim=1).cpu().numpy()     # classe de maior logit
    y = Y.cpu().numpy()
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0)
    return {"acuracia": float(accuracy_score(y, pred)),
            "precision_macro": float(p), "recall_macro": float(r),
            "f1_macro": float(f1)}


def rodar_mnist(config, dados, epocas, seeds=SEEDS_TREINO):
    resultados, historicos = [], []
    for seed in seeds:
        fixar_seed(seed)
        modelo = criar_mlp(784, 10, config["larguras"],       # 784 e 10
                           p_dropout=config.get("dropout", 0.0)).to(DEVICE)
        hist = treinar(modelo, dados["loader"], dados["val"],
                       nn.CrossEntropyLoss(),                 
                       lr=config["lr"], epocas=epocas,
                       momentum=config.get("momentum", 0.0),
                       weight_decay=config.get("l2", 0.0),
                       l1_lambda=config.get("l1", 0.0))
        resultados.append(metricas_classificacao(modelo, dados["val"]))
        historicos.append(hist)
    media  = {k: float(np.mean([r[k] for r in resultados])) for k in resultados[0]}
    desvio = {k: float(np.std([r[k] for r in resultados]))  for k in resultados[0]}
    return media, desvio, historicos
