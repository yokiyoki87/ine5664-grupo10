import datetime
import numpy as np
import csv

#IDs das funções de ativação
F_RELU     = 0
F_LOGIST   = 1
F_TANH     = 2
F_SOFTPLUS = 3


# Funções de ativação.


def relu(x):
    return max(x,0)

def d_relu(x):
    return int(x>=0)


def logistica(x):
    return 1 / (1+np.exp(-x))

def d_logistica(x):
    logis = logistica(x)
    return logis * (1-logis)


def d_tanh(x):
    tanh = np.tanh(x)
    return 1-tanh*tanh


def softplus(x):
    return np.log(1+np.exp(x))

def d_softplus(x):
    expon = np.exp(x)
    return expon / (1+expon)


# Ids das funções de custo

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
        if type(args[0]) in [list,tuple]:
            self.criar(*args)
    

    def criar(self,*camadas,labels_saida=[]):
        """
        Cria uma rede neural a partir da descrição das camadas

        Cada camada é descrita por uma tupla
        (Número de neurônios, id da função de ativação [padrão=ReLU],
        vieses [padrão=0 para cada], funções de custo [padrão=Erro Quadr])

        Se treinar para regressão, a saída vem do primeiro neurônio da última camada
        """

        # cria nova rede MLP
        self.tamanhos = []
        self.pesos = []
        self.ativacao = []
        self.vieses = []
        self.custos = []
        self.n_camadas = 0

        for i in camadas:
            # cria a nova camada
            self.criar_camada(*i)
        
        # se for classificador, vai ser utilizado na saida
        self.classificacao = labels_saida
    
    def criar_camada(self,tamanho,ativacao=F_RELU,custo=C_QUAD,vies=None):
        "Ver acima a definição de camadas"
        
        if self.n_camadas:
            tamanho_anterior = self.tamanhos[-1]
            self.pesos.append(np.full((tamanho_anterior,tamanho),0.5))
            self.ativacao.append(ativacao)
            if not vies: vies = np.zeros(tamanho)
            self.vieses.append(vies)
            self.custos.append(custo)
        self.tamanhos.append(tamanho)
        self.n_camadas += 1


    def carregar(self,arquivo):
        # TODO
        print("NAO IMPLEMENTADO")
    
    def salvar(self):
        with open("modelos_treinados/"+str(datetime.datetime.now())+".txt","w") as f:
            f.write(str(self.tamanhos[0]))
            for i in range(1,self.n_camadas-1):
                dadocam = f"{self.tamanhos[i]} {self.ativacao[i]} {self.custos[i]}\n"
                vies = " ".join(map(str,self.vieses[i])) + "\n"
                f.write(dadocam)
                f.write(vies)
            for i in self.pesos:
                for linha in i:
                    f.write(" ".join(map(str,linha))+"\n")


    def treinar(self,dataset,epocas,taxa_apr,func_erro=C_QUAD):
        """Treina a rede no dataset
        Cada dataset é uma tupla com o último valor sendo a saída esperada
        """
        pesos_t = [np.transpose(i) for i in self.pesos]
        #pesos_t.reverse()
        f_erro = erro_quad if func_erro == C_QUAD else \
                entropia_cruzada_binaria if func_erro == C_BCE else \
                entropia_cruzada_categorica
        for ep in range(epocas):
            for instancia in dataset:
                #feedforward
                real = instancia[-1]
                entrada = instancia[:-1]
                predicao = self.responder_todas(entrada)

                # backpropagation
                
                # determinar erro da ultima camada dependendo do tipo
                tam = np.size(predicao[-1])
                if tam == 1:
                    erro = [np.array(f_erro(predicao[-1],real))]
                else:
                    e = np.ones((tam))
                    e[real] = 0.0
                    erro = [e]

                #calcular erro para cada camada anterior
                for i in range(self.n_camadas-2,0,-1):
                    if self.ativacao[i] == F_RELU:
                        d = np.vectorize(d_relu)
                    elif self.ativacao[i] == F_LOGIST:
                        d = np.vectorize(d_logistica)
                    elif self.ativacao[i] == F_TANH:
                        d = np.vectorize(d_tanh)
                    elif self.ativacao[i] == F_SOFTPLUS:
                        d = np.vectorize(d_softplus)
                    
                    dxde = d(predicao[i])                   #
                    dxdw = np.matmul(erro[-1],pesos_t[i])   # TODO: PEGAR A EQUACAO CERTA
                    prox_erro = np.multiply(dxde,dxdw)      #
                    erro.append(prox_erro)
                erro.reverse()

                for i in range(self.n_camadas-1):
                    for j in range(self.pesos[i].shape[0] if self.pesos[i].shape[1] > 1 else 1):
                        self.pesos[i][j] -= taxa_apr * erro[i][j]
                
                # TODO: continuar aqui implementacao backpropagation
            print(f"epoch {ep} done")


    def testar(self,dataset):

        total = 0
        corretos = 0
        for instancia in dataset:
            total += 1
            res = self.responder(instancia[:-1])
            if res == instancia[-1]:
                corretos += 1
        return corretos/total


    def responder_todas(self,entrada):
        """
        Calcula valor para entrada, dados pesos vieses
        Retorna: Lista de resultados de cada camada
        """
        respostas = []
        anterior = entrada
        for i in range(self.n_camadas-1):
            # determinar funcao de ativacao desta camada
            if self.ativacao[i] == F_RELU:
                f = np.vectorize(relu)
            elif self.ativacao[i] == F_LOGIST:
                f = np.vectorize(logistica)
            elif self.ativacao[i] == F_TANH:
                f = np.tanh
            elif self.ativacao[i] == F_SOFTPLUS:
                f = np.vectorize(softplus)
            
            # determinar os resultados

            axb = np.add(np.matmul(anterior,self.pesos[i]),self.vieses[i])
            saida = f(axb)
            respostas.append(saida)
            anterior = saida
        return respostas

    def responder(self,entrada):
        resultado = self.responder_todas(entrada)[-1]
        if np.size(resultado) == 1:
            return resultado[0]
        if self.classificacao:
            return self.classificacao[np.argmax(resultado)]
        return np.argmax(resultado)



def abrir_dataset(arq,delimita=",",prop=0.9):
    "Abre o dataset .csv, removendo a legenda da primeira fileira"
    dados = []
    with open(arq) as f:
        raw = csv.reader(f,dialect='excel',delimiter=delimita)
        dados = [i for i in raw]
    dados.pop(0) # tira labels do dataset
    dados = [list(map(float,i)) for i in dados]
    tam = int(len(dados)*prop)
    treino = dados[:tam]
    teste = dados[tam:]
    return treino,teste

if __name__ == "__main":
    print("this is the only way it could have ended")