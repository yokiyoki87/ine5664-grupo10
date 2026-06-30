# ine5664-grupo10
Implementação de uma biblioteca Python de Redes Neurais Multicamadas (MLP)

por Calipso Youko Wada (20100540) e Marcos Henrique Kogler (21250087)

[Ainda em desenvolvimento e correção]

---
# Utilização
# 1. Criar uma nova rede
Crie uma rede com RedeMulticamadas(*camadas)
Cada camada é definida como uma tupla:

(Tamanho da Camada, Função de Ativação, Função de Custo, Vieses da Camada)

Tamanho da Camada : int = Número de neurônios na camada

Função de Ativação : int = escolher entre F_RELU, F_LOGIST (logística), F_TANH, e F_SOFTPLUS

Função de Custo : int = escolher entre C_QUAD (erro quadrático), C_BCE (entropia cruzada binária), e C_CCE (entropia cruzada categórica)

Vieses da camada : list[float] = lista ou vetor de vieses da camada

# 2. Treinamento
Utilize um conjunto de dados para treinar a rede
O arquivo deve estar em formato .csv, com a primeira fileira tendo os nomes das colunas, e a última coluna tendo os valores de saída
Carregue os conjuntos treino e teste com abrir_dataset(nome do arquivo, separador=',', proporção treino/teste = 0.9)
treine com RedeMulticamadas.treinar(conjunto de treino, epochs, taxa de aprendizado,função de custo) [FALTAM CORREÇÕES NO MÉTODO]

# 3. Teste
Com RedeMulticamadas.testar(conjunto de teste), pode-se obter a acurácia ou erro médio, dependendo de se a rede era de regressão [NÃO IMPLEMENTADO AINDA] ou classificação

# 1A. Carregar rede treinada previamente
Após treinar uma rede, ela pode ser salva como "datahora.txt" na pasta modelos_treinados/ com RedeMulticamadas.salvar() para posteriormente ser carregada com RedeMulticamadas.carregar(nome do arquivo) [NÃO IMPLEMENTADO AINDA]