import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED_DADOS = 0 # aqui temos que fixar a semente dos dados
SEEDS_TREINO = [1,2,3] # para variar o init dos pesos e ordem dos batches

def fixar_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def gerar_dados(n=300, seed=SEED_DADOS):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n) # valores aleatórios entre -5 e 5
    ruido = rng.normal(0, 0.2, size=n)
    y = np.sin(3*x) + (0.3*x) + (1.5 *(x>2)) + ruido 
    return x, y

def dividir(x, y, p_treino=0.70, p_val=0.15, seed=SEED_DADOS):
    rng = np.random.default_rng(seed + 1000)
    idx = rng.permutation(len(x))
    n_tr, n_va = int(p_treino*len(x)), int(p_val*len(x))
    i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return (x[i_tr], y[i_tr]), (x[i_va], y[i_va]), (x[i_te], y[i_te])

class Padronizador:
    """ Média e desvio do treino, o conjunto todo resulta em vazamento """

    def fit(self, v):
        self.mu, self.sd = float(v.mean()), float(v.std())
        return self
    def transform(self, v):
        return (v - self.mu) / self.sd
    def inverse(self, v):
        return v * self.sd + self.mu

def preparar(n=300, batch_size=32):
    x, y = gerar_dados(n)
    (xtr, ytr), (xva, yva), (xte, yte) = dividir(x, y)

    px, py = Padronizador().fit(xtr), Padronizador().fit(ytr)

    def T(v):
        return torch.tensor(v, dtype=torch.float32).view(-1, 1)
    
    Xtr, Xva, Xte = T(px.transform(xtr)), T(px.transform(xva)), T(px.transform(xte))
    Ytr, Yva, Yte = T(py.transform(ytr)), T(py.transform(yva)), T(py.transform(yte))

    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    return {"loader": loader, "val": (Xva, Yva), "teste": (Xte, Yte),
            "treino": (Xtr, Ytr), "py": py, "px": px, "bruto": (x, y)}

def criar_mlp(n_entrada, n_saida, larguras, p_dropout=0.0):
    camadas, anterior = [], n_entrada
    for largura in larguras:
        camadas += [nn.Linear(anterior, largura), nn.ReLU()]
        if p_dropout > 0:
            camadas.append(nn.Dropout(p_dropout))
        anterior = largura
    camadas.append(nn.Linear(anterior, n_saida))   #saída linear
    return nn.Sequential(*camadas)


@torch.no_grad()
def avaliar(modelo, dados, loss_fn):
    modelo.eval()
    X, Y = dados[0].to(DEVICE), dados[1].to(DEVICE)
    return loss_fn(modelo(X), Y).item()

def treinar(modelo, loader, val, loss_fn, lr, epocas=300,
            momentum=0.0, weight_decay=0.0, l1_lambda=0.0):
    otim = torch.optim.SGD(modelo.parameters(), lr=lr,
                           momentum=momentum, weight_decay=weight_decay)
    hist = {"treino": [], "val": []}

    for _ in range(epocas):
        modelo.train()
        soma, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            otim.zero_grad()

            perda_dados = loss_fn(modelo(xb), yb)
            perda = perda_dados
            if l1_lambda > 0:                       #L1 não existe no SGD, aqui tive que fazer na mão
                l1 = sum(p.abs().sum() for p in modelo.parameters() if p.dim() > 1)
                perda = perda + l1_lambda * l1

            perda.backward()
            otim.step()
            soma += perda_dados.item() * len(xb)    #registro de perda sem penalidade
            n += len(xb)

        hist["treino"].append(soma / n)
        hist["val"].append(avaliar(modelo, val, loss_fn))
    return hist


@torch.no_grad()
def metricas_regressao(modelo, dados, py):
    modelo.eval()
    X, Y = dados[0].to(DEVICE), dados[1].to(DEVICE)
    y_pred = py.inverse(modelo(X).cpu())            #na escala de f(x)
    y_true = py.inverse(Y.cpu())
    erro = y_true - y_pred
    mse = (erro ** 2).mean()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {"MAE": erro.abs().mean().item(), "MSE": mse.item(),
            "RMSE": mse.sqrt().item(),
            "R2": (1 - (erro ** 2).sum() / ss_tot).item()}


def rodar(config, dados, epocas=300, seeds=SEEDS_TREINO):
    resultados, historicos = [], []
    for seed in seeds:
        fixar_seed(seed)
        modelo = criar_mlp(1, 1, config["larguras"],
                           p_dropout=config.get("dropout", 0.0)).to(DEVICE)
        hist = treinar(modelo, dados["loader"], dados["val"], nn.MSELoss(),
                       lr=config["lr"], epocas=epocas,
                       momentum=config.get("momentum", 0.0),
                       weight_decay=config.get("l2", 0.0),
                       l1_lambda=config.get("l1", 0.0))
        resultados.append(metricas_regressao(modelo, dados["val"], dados["py"]))
        historicos.append(hist)

    media = {k: float(np.mean([r[k] for r in resultados])) for k in resultados[0]}
    desvio = {k: float(np.std([r[k] for r in resultados])) for k in resultados[0]}
    return media, desvio, historicos



