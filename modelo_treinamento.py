from modelo_definicao import *


dataset_arqs = [
    "datasets/BudgetFood.csv"
]

for arq in dataset_arqs:
    treino,teste = abrir_dataset(arq)
    modelo = RedeMulticamadas(
        (4,F_RELU),
        (5,F_LOGIST),
        (5,F_LOGIST),
        (1,F_TANH))
    modelo.treinar(treino,10,0.1)
    modelo.testar(teste)
    modelo.salvar()
