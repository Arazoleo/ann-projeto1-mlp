import json
from mnist import preparar_mnist, rodar_mnist

EPOCAS = 60
BASE = {"larguras": [256, 256], "lr": 0.1}

ABLACOES = {
    "L1":       ("l1",       [1e-6, 1e-5, 1e-4]),
    "L2":       ("l2",       [1e-5, 1e-4, 1e-3]),
    "dropout":  ("dropout",  [0.1, 0.2, 0.5]),
    "momentum": ("momentum", [0.5, 0.9, 0.99]),
}

if __name__ == "__main__":
    dados = preparar_mnist()

    print("Baseline")
    m_base, d_base, _ = rodar_mnist(BASE, dados, epocas=EPOCAS)
    print(f"  acc={m_base['acuracia']:.4f} ± {d_base['acuracia']:.4f}"
          f"  F1={m_base['f1_macro']:.4f}\n")

    resultados = {"baseline": {"config": BASE, "media": m_base, "desvio": d_base}}

    for nome, (chave, valores) in ABLACOES.items():
        print(nome)
        tabela = []
        for v in valores:
            cfg = dict(BASE)          # sempre parte do baseline, nunca acumula
            cfg[chave] = v
            media, desvio, _ = rodar_mnist(cfg, dados, epocas=EPOCAS)
            tabela.append({"valor": v, **media, "acc_std": desvio["acuracia"]})
            print(f"  {chave}={v:<8g} acc={media['acuracia']:.4f} ± "
                  f"{desvio['acuracia']:.4f}  F1={media['f1_macro']:.4f}"
                  f"  (Δ={media['acuracia'] - m_base['acuracia']:+.4f})")
        melhor = max(tabela, key=lambda r: r["acuracia"])
        print(f"  -> melhor {chave}={melhor['valor']}"
              f"  Δacc = {melhor['acuracia'] - m_base['acuracia']:+.4f}\n")
        resultados[nome] = {"tabela": tabela, "melhor": melhor}

    json.dump(resultados, open("ablacoes_mnist.json", "w"), indent=2, default=str)
