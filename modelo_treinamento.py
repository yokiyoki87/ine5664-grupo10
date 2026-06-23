from modelo_definicao import *

modelo = RedeMulticamadas()
dataset_arqs = [
    "the_break",
    "the_shattering_circle",
    "event_horizon"
]

for arq in dataset_arqs:
    dataset = abrir_dataset()
    modelo.treinar(dataset)
    modelo.testar()
    modelo.salvar()
