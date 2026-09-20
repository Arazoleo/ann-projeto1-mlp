import json
import numpy as np
from mnist import preparar_mnist, rodar_mnist

EPOCAS = 60


def buscar(eixo, candidatos, base, dados, epocas=EPOCAS):
    """Varre um eixo com o resto congelado. Decide pela acurácia de validação."""
    tabela = []
    for valor in candidatos:
        cfg = dict(base)
        cfg[eixo] = valor
        media, desvio, _ = rodar_mnist(cfg, dados, epocas=epocas)
        tabela.append({"valor": valor, **media, "acc_std": desvio["acuracia"]})
        print(f"  {eixo}={str(valor):16s} acc={media['acuracia']:.4f} ± "
              f"{desvio['acuracia']:.4f}  F1={media['f1_macro']:.4f}  "
              f"recall={media['recall_macro']:.4f}")
    melhor = max(tabela, key=lambda r: r["acuracia"])
    print(f"  -> melhor: {eixo}={melhor['valor']}\n")
    return melhor["valor"], tabela


if __name__ == "__main__":
    dados = preparar_mnist()
    base = {"larguras": [128], "lr": 0.05}
    registro = {}

    print("Eixo 1: largura (uma camada oculta)")
    base["larguras"], registro["largura"] = buscar(
        "larguras", [[64], [128], [256]], base, dados)

    w = base["larguras"][0]
    print("Eixo 2: profundidade")
    base["larguras"], registro["profundidade"] = buscar(
        "larguras", [[w], [w, w], [w, w, w]], base, dados)

    print("Eixo 3: taxa de aprendizado")
    base["lr"], registro["lr"] = buscar("lr", [0.01, 0.05, 0.1], base, dados)

    print("Baseline MNIST escolhido:", base)
    json.dump({"baseline": base, "busca": registro},
              open("baseline_mnist.json", "w"), indent=2, default=str)
