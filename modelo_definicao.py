
import numpy as np

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
            self.criar(args)
    

    def criar(self,*camadas,labels_saida=[]):
        """
        Cria uma rede neural a partir da descrição das camadas

        Cada camada é descrita por uma tupla
        (Número de neurônios, id da função de ativação [padrão=ReLU],
        vieses [padrão=0 para cada], funções de custo [padrão=Erro Quadr])

        Se treinar para regressão, a saída vem do primeiro neurônio da última camada
        """

        # cria nova rede MLP
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
    
    def criar_camada(self,tamanho,ativacao=F_RELU,vies=None,custo=C_QUAD):
        "Ver acima a definição de camadas"

        tamanho_anterior = self.pesos[-1] if self.pesos else 1
        self.pesos.append(np.full((tamanho_anterior,tamanho),0.5))
        self.ativacao.append(ativacao)
        if not vies: vies = np.zeros(tamanho)
        self.vieses.append(vies)
        self.custos.append(custo)
        self.n_camadas += 1


    def carregar(self,*args):
        # TODO
        pass
    
    def salvar(self,*args):
        # TODO
        pass


    def treinar(self,dataset,epocas,taxa_apr,func_erro=C_QUAD):
        """Treina a rede no dataset
        Cada dataset é uma tupla com o último valor sendo a saída esperada
        """
        pesos_t = [np.transpose(i) for i in self.pesos]
        f_erro = erro_quad if func_erro == C_QUAD else \
                entropia_cruzada_binaria if func_erro == C_BCE else \
                entropia_cruzada_categorica
        for _ in range(epocas):
            for instancia in dataset:
                #feedforward
                real = instancia[-1]
                entrada = instancia[:-1]
                predicao = self.responder_todos(entrada)

                # backpropagation
                
                # determinar erro da ultima camada dependendo do tipo
                tam = np.size(predicao[-1])
                if tam == 1:
                    erro = [np.array([f_erro(predicao,entrada)])]
                else:
                    e = np.ones((tam))
                    e[real] = 0.0
                    erro = [e]

                #calcular erro para cada camada anterior
                for i in range(self.n_camadas-1,-1,-1):
                    if self.ativacao == F_RELU:
                        d = np.vectorize(d_relu)
                    elif self.ativacao == F_LOGIST:
                        d = np.vectorize(d_logistica)
                    elif self.ativacao == F_TANH:
                        d = np.vectorize(d_tanh)
                    elif self.ativacao == F_SOFTPLUS:
                        d = np.vectorize(d_softplus)
                    prox_erro = np.multiply(d(predicao[i]),np.matmul(erro[-1],pesos_t[i]))
                    erro.append(prox_erro)
                erro = erro.reverse()

                for i in range(self.n_camadas):
                    for j in range(len(erro[i])):
                        self.pesos[i][j] -= taxa_apr * erro[i][j]
                
                # TODO: continuar aqui implementacao backpropagation

                    


    def testar_acuracia(self,dataset):
        total = 0
        corretos = 0
        for instancia in dataset:
            total += 1
            res = self.responder(instancia)
            if res == instancia[-1]:
                corretos += 1
        return total / corretos


    def responder_todas(self,entrada):
        """
        Calcula valor para entrada, dados pesos vieses
        Retorna: Lista de resultados de cada camada
        """
        respostas = []
        anterior = entrada
        for i in range(self.n_camadas):
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



def abrir_dataset(arq):
    with open(arq) as f:
        dados = f.readlines()
    
    # TODO: processar dados

    return dados

if __name__ == "__main":
    print("this is the only way it could have ended")