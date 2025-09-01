En este trabajo se buscó modelar un sistema de clasificación de imágenes manuscritas
japonesas utilizando redes neuronales. Para llevarlo a cabo, se implementaron distintos mo-
delos tanto manualmente como con PyTorch, explorando múltiples arquitecturas, técnicas
de regularización (como L2 y early stopping) y optimizadores (incluyendo Adam y SGD
con mini-batch). Se desarrollaron modelos M0 a M4, donde se aplicaron estrategias de
evaluación para comparar el impacto de cada mejora sobre el rendimiento y la capacidad
de generalización.

Utilizando el mejor modelo, se generaron las predicciones a posteriori para un conjunto no etiquetado y se almacenaron en
un archivo Amblard_Agustin_predicciones.csv
