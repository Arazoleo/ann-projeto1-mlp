import json
import numpy as np
from setup import preparar, rodar
import matplotlib.pyplot as plt

EPOCAS = 10000

def buscar(eixo, candidatos, base, dados, epocas=EPOCAS):
    """Varre UM eixo com todo o resto congelado. Decide pela validação."""
    tabela = []
    for valor in candidatos:
        cfg = dict(base)          #a base só muda quando o eixo termina
        cfg[eixo] = valor
        media, desvio, _ = rodar(cfg, dados, epocas=epocas)
        tabela.append({"valor": valor, **media, "R2_std": desvio["R2"]})
        print(f"  {eixo}={str(valor):10s} R2={media['R2']:.4f} ± {desvio['R2']:.4f}"
              f"  RMSE={media['RMSE']:.4f}  MAE={media['MAE']:.4f}")
    melhor = max(tabela, key=lambda r: r["R2"])
    print(f"  -> melhor: {eixo}={melhor['valor']}\n")
    return melhor["valor"], tabela



def plotar_curvas(hist, titulo):
    ep_min = int(np.argmin(hist["val"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["treino"], label="treino")
    ax.plot(hist["val"], label="validação")
    ax.axvline(ep_min, ls="--", c="gray", label=f"mín. val (época {ep_min})")
    ax.set_xlabel("época"); ax.set_ylabel("MSE (escala padronizada)")
    ax.set_yscale("log")          # sem log você não enxerga o fim do treino
    ax.set_title(titulo); ax.legend()
    return fig



if __name__ == "__main__":
    dados = preparar(n=300)
    base = {"larguras": [64], "lr": 0.05}
    registro = {}

    print("Eixo 1: largura (uma camada oculta)")
    base["larguras"], registro["largura"] = buscar(
        "larguras", [[16], [32], [64], [128]], base, dados)

    w = base["larguras"][0]
    print("Eixo 2: profundidade")
    base["larguras"], registro["profundidade"] = buscar(
        "larguras", [[w], [w, w], [w, w, w]], base, dados)

    print("Eixo 3: taxa de aprendizado")
    base["lr"], registro["lr"] = buscar("lr", [0.01, 0.05, 0.1], base, dados)

    print("Baseline escolhido:", base)
    json.dump({"baseline": base, "busca": registro},
              open("baseline.json", "w"), indent=2, default=str)
