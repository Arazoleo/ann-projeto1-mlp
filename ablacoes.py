import json
from setup import preparar, rodar  

EPOCAS = 10000
BASE = {"larguras": [128, 128], "lr": 0.05}

ABLACOES = {
    "L1":       ("l1",       [1e-6, 1e-5, 1e-4]),
    "L2":       ("l2",       [1e-5, 1e-4, 1e-3]),
    "dropout":  ("dropout",  [0.1, 0.2, 0.5]),
    "momentum": ("momentum", [0.5, 0.9, 0.99]),
}

if __name__ == "__main__":
    dados = preparar(n=300)

    print("Baseline")
    m_base, d_base, _ = rodar(BASE, dados, epocas=EPOCAS)
    print(f"  R2={m_base['R2']:.4f} ± {d_base['R2']:.4f}"
          f"  RMSE={m_base['RMSE']:.4f}  MAE={m_base['MAE']:.4f}\n")

    resultados = {"baseline": {"config": BASE, "media": m_base, "desvio": d_base}}

    for nome, (chave, valores) in ABLACOES.items():
        print(nome)
        tabela = []
        for v in valores:
            cfg = dict(BASE)          #sempre partir do baseline, nunca acumular
            cfg[chave] = v
            media, desvio, _ = rodar(cfg, dados, epocas=EPOCAS)
            tabela.append({"valor": v, **media, "R2_std": desvio["R2"]})
            print(f"  {chave}={v:<8g} R2={media['R2']:.4f} ± {desvio['R2']:.4f}"
                  f"  (Δ={media['R2'] - m_base['R2']:+.4f})")
        melhor = max(tabela, key=lambda r: r["R2"])
        print(f"  -> melhor {chave}={melhor['valor']}"
              f"  ΔR² = {melhor['R2'] - m_base['R2']:+.4f}\n")
        resultados[nome] = {"tabela": tabela, "melhor": melhor}

    json.dump(resultados, open("ablacoes.json", "w"), indent=2, default=str)
