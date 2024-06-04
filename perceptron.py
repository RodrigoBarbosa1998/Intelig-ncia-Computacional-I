import numpy as np
import random
from numpy import sign as sinal
import matplotlib.pyplot as plt


class PerceptronModel:
    def __init__(self, num_pontos, num_subset=10, iteracoes=1000):
        self.num_pontos = num_pontos

        # Cria uma linha aleatória como função alvo f(x)
        ponto1 = [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        ponto2 = [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]

        x1A = ponto1[1]
        x2A = ponto1[2]

        x1B = ponto2[1]
        x2B = ponto2[2]

        # Cria os pesos da função alvo com base nos pontos aleatórios
        self.pesos_alvo = np.array([x1B * x2A - x1A * x2B, x2B - x1A, x1A - x1B])

        # Inicializa o conjunto de treinamento e seus rótulos corretos
        self.conjunto_treino, self.rotulos = self.gera_conjunto_treino()

        # Inicializa o conjunto fora da amostra
        self.conjunto_teste = np.random.uniform(-1, 1, size=(10000, 2))
        x0 = np.ones((10000, 1))
        self.conjunto_teste = np.insert(self.conjunto_teste, [0], x0, axis=1)

        # Inicializa o melhor perceptron
        self.melhor_perceptron = None

        # Executa o modelo com um subconjunto do conjunto de treinamento
        self.executa(num_subset, iteracoes)

        # Executa o modelo com o conjunto de dados completo
        self.executa(num_pontos, iteracoes)

    @staticmethod
    def aplicar(pesos, ponto):
        # Aplica h(x)
        return sinal(np.dot(pesos, ponto))

    @staticmethod
    def aprender(pesos, ponto, rotulo):
        # Aprende a partir das classificações erradas
        return pesos + rotulo * ponto
    
    def gera_conjunto_treino(self):
        """Gera um conjunto de dados uniformemente distribuído de tamanho n por 3"""

        # Gera pontos uniformemente distribuídos aleatórios
        conjunto_treino = np.random.uniform(-1, 1, size=(self.num_pontos, 2))

        # Insere valores x0 no início da matriz
        x0 = np.ones((self.num_pontos, 1))
        conjunto_treino = np.insert(conjunto_treino, [0], x0, axis=1)

        # Gera rótulos para o conjunto de treinamento
        rotulos = []
        [rotulos.append(PerceptronModel.aplicar(self.pesos_alvo, ponto)) for ponto in conjunto_treino]
        rotulos = np.array([rotulos]).T

        # Define o conjunto de treinamento e os rótulos como atributos da instância
        return conjunto_treino, rotulos

    def plota(self, tam_conjunto):
        # Plota o conjunto com base no tamanho do conjunto fornecido
        rotulos = self.rotulos[0:tam_conjunto]
        conjunto_treino = self.conjunto_treino[0:tam_conjunto]

        # Ajusta o eixo para caber no conjunto de dados
        plt.axis([-1, 1, -1, 1])

        # Separa o conjunto de dados em dois arrays, cada um representando seu rótulo por f(x)
        conjunto_acima_linha = []
        conjunto_abaixo_linha = []

        for i in range(len(rotulos)):
            if rotulos[i] == 1:
                conjunto_acima_linha.append(conjunto_treino[i])
            else:
                conjunto_abaixo_linha.append(conjunto_treino[i])

        conjunto_acima_linha = np.array(conjunto_acima_linha)
        conjunto_abaixo_linha = np.array(conjunto_abaixo_linha)

        # Plota os conjuntos
        if conjunto_acima_linha.size > 0:
            x1_acima = conjunto_acima_linha[:, 1]
            x2_acima = conjunto_acima_linha[:, 2]
            plt.scatter(x1_acima, x2_acima, c='b')

        if conjunto_abaixo_linha.size > 0:
            x1_abaixo = conjunto_abaixo_linha[:, 1]
            x2_abaixo = conjunto_abaixo_linha[:, 2]
            plt.scatter(x1_abaixo, x2_abaixo, c='r')

        # Gera 50 números espaçados uniformemente de -1 a 1
        linha = np.linspace(-1, 1)

        # Plota a f(x) em azul
        m, b = -self.pesos_alvo[1] / self.pesos_alvo[2], -self.pesos_alvo[0] / self.pesos_alvo[2]
        plt.plot(linha, m * linha + b, 'b-', label='f(x)')

        # Plota a g(x) em vermelho tracejado
        m1, b1 = -self.melhor_perceptron[1] / self.melhor_perceptron[2], -self.melhor_perceptron[0] / self.melhor_perceptron[2]
        plt.plot(linha, m1 * linha + b1, 'r--', label='h(x)')

        plt.show()

    def pla(self, tam_conjunto):
        """Algoritmo de Aprendizado do Perceptron"""
        # Toma um subconjunto do conjunto de dados com base no tamanho fornecido
        conjunto_treino = self.conjunto_treino[0:tam_conjunto]
        rotulos_corretos = self.rotulos[0:tam_conjunto]

        def aplica_h(pesos):
            # Classifica o conjunto de dados usando os pesos da hipótese
            rotulos_pla = []
            for i in range(len(conjunto_treino)):
                rotulos_pla.append(PerceptronModel.aplicar(pesos, conjunto_treino[i]))
            return rotulos_pla

        def misclassificados(rotulos_pla):
            # Retorna uma lista de índices de pontos classificados erroneamente com base na última hipótese encontrada
            misclassificados = []
            for i in range(tam_conjunto):
                if self.rotulos[i] != rotulos_pla[i]:
                    misclassificados.append(i)
            return misclassificados

        # Inicia o vetor de pesos com zeros
        pesos = [0, 0, 0]

        # Aplica o algoritmo ao conjunto de dados com pesos iniciais iguais a 0
        # Todos os pontos são inicialmente classificados erroneamente
        rotulos_pla = aplica_h(pesos)

        # Gera o conjunto de pontos classificados erroneamente
        indices_errados = misclassificados(rotulos_pla)

        total_iteracoes = 1
        total_incorretos = len(indices_errados)

        while len(indices_errados) > 0:
            # Escolhe um índice aleatório para aprender
            indice_aleatorio = random.choice(indices_errados)
            pesos = PerceptronModel.aprender(pesos, conjunto_treino[indice_aleatorio], rotulos_corretos[indice_aleatorio])

            # Aplica o algoritmo com os pesos atualizados
            rotulos_pla = aplica_h(pesos)

            # Gera uma nova lista de índices classificados erroneamente
            indices_errados = misclassificados(rotulos_pla)

            total_iteracoes += 1
            total_incorretos += len(indices_errados)

        # melhor_perceptron tem a menor diferença
        diferenca = self.avalia_diferenca(pesos)
        if self.melhor_perceptron is None or diferenca < self.melhor_perceptron[3]:
            self.melhor_perceptron = np.hstack((pesos, diferenca))

        return total_iteracoes, total_incorretos

    def avalia_diferenca(self, hipotese):
        # Avalia f e g em uma amostra de 10000 pontos fora da amostra
        amostra = self.conjunto_teste
        tamanho_amostra = len(amostra)

        i = 0
        total_errados = 0
        while i < tamanho_amostra:
            ponto = amostra[i]
            classificacao_alvo = PerceptronModel.aplicar(self.pesos_alvo, ponto)
            classificacao_hipotese = PerceptronModel.aplicar(hipotese, ponto)
            if classificacao_alvo != classificacao_hipotese:
                total_errados += 1
            i += 1

        return total_errados / tamanho_amostra

    @staticmethod
    def exibe_resultados(tam_conjunto, media_iteracoes, diferenca):
        print(f'Média de iterações para convergir para o conjunto de dados N={tam_conjunto}: {media_iteracoes}')
        print(f'Média P[f(x) != g(x)]: {diferenca}')

    def executa(self, tamanho, iteracoes):
        print(f'Executando para conjunto de dados N={tamanho}, Iterações= {iteracoes} vezes, e avaliando cada hipótese contra um conjunto de teste de 10.000 pontos.')

        # Reinicializa o melhor_perceptron
        self.melhor_perceptron = None

        total_iteracoes = 0
        i = 0
        while i < iteracoes:
            resultado = self.pla(tamanho)
            total_iteracoes += resultado[0]
            i += 1
        media_iteracoes = total_iteracoes / iteracoes

        # Avalia a diferença entre f e g em um dado fora da amostra
        diferenca = self.avalia_diferenca(self.melhor_perceptron[0:3])

        self.plota(tamanho)
        PerceptronModel.exibe_resultados(tamanho, media_iteracoes, diferenca)

# Instancia e executa o modelo Perceptron
modelo = PerceptronModel(100, num_subset=10, iteracoes=1000)
