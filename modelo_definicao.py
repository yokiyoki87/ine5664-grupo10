
import numpy as np

#IDs das funções
F_LIMIAR   = 0
F_IDENT    = 1
F_RELU     = 2
F_LOGIST   = 3
F_TANH     = 4
F_SOFTMAX  = 5
F_SOFTPLUS = 6


# Funções de ativação.

def logistica(x):
    return 1 / (1+np.exp(-x))

def d_logistica(x):
    logis = logistica(x)
    return logis * (1-logis)


def d_tanh(x):
    tanh = np.tanh(x)
    return 1-tanh*tanh


def limiar(x,t):
    return x >= t


def relu(x):
    return max(x,0)


def softmax(x,valores):
    return np.exp(x) / np.sum(np.exp(valores))

def softplus(x):
    return np.log(1+np.exp(x))

def d_softplus(x):
    expon = np.exp(x)
    return expon / (1+expon)


C_QUAD = 0
C_BCE  = 1
C_CCE  = 2

def erro_quad(y1,y2):
    # ideal para regressao
    return (y2-y1)*(y2-y1)

def entropia_cruzada_binaria(y1,y2):
    # ideal para 2 classes
    return -(y1*np.log(y2)+(1-y1)*np.log(1-y2))

def entropia_cruzada_categorica(my1,my2):
    # ideal para n>2 classes
    return -np.sum(np.multiply(my1,np.log(my2)))


class RedeMulticamadas:
    def __init__(self,*args):
        if type(args[0]) is int:
            self.criar(args)
        elif type(args[0]) in [list,tuple]:
            self.carregar(args)
    

    def criar(self,n_entradas,*camadas,labels_saida=[]):
        # cria nova rede MLP
        self.pesos = []
        self.f_ativacao = []
        self.vieses = []

        tamanho_camada_anterior = n_entradas
        for i,cam in enumerate(camadas):
            # cria a nova camada
            self.pesos.append(np.full((tamanho_camada_anterior,cam[0]),0.5))
            self.f_ativacao.append(cam[1])
            tamanho_camada_anterior = cam[0]
        
        # se for classificador, vai ser utilizado na saida
        self.classificacao = labels_saida
    
    def carregar(self,*args):
        pass

    def treinar(self,*args):
        pass

    def testar(self,*args):
        pass

    def responder(self,*args):
        pass
    
    def salvar(self,*args):
        pass



def abrir_dataset(arq):
    with open(arq) as f:
        dados = f.readlines()
    
    # processar dados

    return dados

if __name__ == "__main":
    print("this is the only way it could have ended")