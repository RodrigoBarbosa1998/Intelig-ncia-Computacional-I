import numpy as np
from sklearn.linear_model import LinearRegression as SKLinearRegression
from numpy.linalg import inv
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.weights = None
    
    # Função para criar vetor de pesos para f(x)
    def criar_pesos_fx(self, pontos):
        reg = SKLinearRegression()
        x_vector = pontos[:, 0].reshape(-1, 1)
        y_vector = pontos[:, 1].reshape(-1, 1)
        reg.fit(x_vector, y_vector)
        w0_f = reg.intercept_[0]
        w1_f = reg.coef_[0][0]
        return np.array((w0_f, w1_f))
    
    # Função para criar pontos e rótulos aleatórios
    def criar_pontos_e_rotulos(self, N, pesos_fx):
        X = np.random.uniform(-1, 1, (N, 2))
        Y = X[:, 1] - X[:, 0] * pesos_fx[1] - pesos_fx[0] >= 0
        Y = np.where(Y, 1, -1)
        return X, Y
    
    # Função para executar regressão linear
    def regressao_linear(self, N=100):
        pontos_aleatorios = np.random.uniform(-1, 1, (2, 2))
        pesos_fx = self.criar_pesos_fx(pontos_aleatorios)
        X, Y = self.criar_pontos_e_rotulos(N, pesos_fx)
        
        X_extendido = np.hstack((np.ones((N, 1)), X))
        Y_extendido = np.expand_dims(Y, 1)
        
        w_pseudo = np.dot(inv(np.dot(X_extendido.T, X_extendido)), X_extendido.T)
        w_gx_novo = np.dot(w_pseudo, Y_extendido)
        
        w_gx = -w_gx_novo[:2] / (w_gx_novo[2] + 1e-10)  
        # Adiciona uma pequena quantidade ao denominador (fugir do erro de dividor por zero)
        # Esse aviso de RuntimeWarning: invalid value encountered in divide indica que houve uma tentativa de dividir por zero ou que o resultado da divisão foi um valor indefinido, resultando em um valor inválido. 
        # Isso geralmente ocorre quando w_gx_novo[2] é zero.
        # Isso pode acontecer se os pontos de dados forem linearmente separáveis, o que faz com que w_gx_novo[2] seja zero, já que estamos dividindo por um dos coeficientes do vetor de pesos.
        
        Y_hat = np.where(np.dot(X_extendido, w_gx_novo) > 0, 1, -1)
        E_in = np.mean(Y_hat != np.expand_dims(Y, 1))
        return pesos_fx, w_gx, X, Y, w_gx_novo, X_extendido, Y_extendido, E_in
    
    # Função para plotar pontos e linhas
    def plotar_pontos_e_linhas(self, pesos_fx, pesos_gx, X, Y, E_in):
        linha = np.linspace(-1, 1, 1001)
        
        plt.plot(linha, linha * pesos_fx[1] + pesos_fx[0], label="f(x)")
        plt.plot(linha, linha * pesos_gx[1] + pesos_gx[0], label="g(x)")
        
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        plt.title("E_in = {}".format(round(E_in, 2)))
        
        plt.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], marker="+", c="r", label="+")
        plt.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], marker="_", c="b", label="-")
        plt.legend(loc="best")
        plt.show()
    
    # Função para realizar múltiplas execuções de regressão linear e calcular E_in médio
    def experimento(self, N=100, tentativas=1000):
        w_fs = []
        w_gs = []
        E_ins = np.zeros(tentativas)
        
        for _ in range(tentativas):
            pesos_fx, w_gx, X, Y, w_gx_novo, X_ext, Y_ext, E_in = self.regressao_linear(N)
            w_fs.append(pesos_fx)
            w_gs.append(w_gx_novo)
            E_ins[_] = E_in
        
        return w_fs, w_gs, np.mean(E_ins)
    
    # Função para calcular E_out médio
    def experimento2(self, w_fs, w_gs, N=25):
        E_outs = np.zeros(len(w_fs))
        
        for i, (pesos_fx, w_gx_novo) in enumerate(zip(w_fs, w_gs)):
            X, Y = self.criar_pontos_e_rotulos(N, pesos_fx)
            X_extendido = np.hstack((np.ones((N, 1)), X))
            Y_extendido = np.expand_dims(Y, 1)
            Y_hat = np.where(np.dot(X_extendido, w_gx_novo) > 0, 1, -1)
            E_out = np.mean(Y_hat != np.expand_dims(Y, 1))
            E_outs[i] = E_out
        
        return np.mean(E_outs)
    
    # Função para criar vetor de pesos para g(x) usando PLA
    def criar_pesos_gx(self, X, Y, w_gx_novo):
        convergiu = False
        n = 0
        w_g = w_gx_novo
        
        while not convergiu:
            convergiu = True
            
            for i in range(len(X)):
                Xi = np.expand_dims(np.hstack((1, X[i])), 0)
                if Y[i] * np.dot(Xi, w_g) < 0:
                    n += 1
                    w_g += Y[i] * Xi.T
                    convergiu = False
        
        return n
    
    # Função para realizar múltiplas execuções de PLA e calcular número médio de iterações
    def experimento3(self, tentativas=1000, N=10):
        iteracoes_por_tentativa = np.zeros(tentativas)
        
        for _ in range(tentativas):
            pesos_fx, w_gx, X, Y, w_gx_novo, X_ext, Y_ext, E_in = self.regressao_linear(N)
            iteracoes = self.criar_pesos_gx(X, Y, w_gx_novo)
            iteracoes_por_tentativa[_] = iteracoes
        
        return np.mean(iteracoes_por_tentativa)
    
    def pocket_pla(self, X, Y, max_iter=1000):
        N, d = X.shape
        w = np.zeros(d)
        w_pocket = np.copy(w)
        erro_pocket = self.calcular_erro(X, Y, w_pocket)
        
        for _ in range(max_iter):
            erro = self.calcular_erro(X, Y, w)
            if erro == 0:
                return w_pocket
            i = np.random.choice(N)
            if np.sign(np.dot(X[i], w)) != Y[i]:
                w_novo = w + Y[i] * X[i]
                erro_novo = self.calcular_erro(X, Y, w_novo)
                if erro_novo < erro_pocket:
                    w_pocket = np.copy(w_novo)
                    erro_pocket = erro_novo
                w = np.copy(w_novo)
        
        return w_pocket
    
    def inverter_rotulos(self, Y, percentual=0.1):
        num_inversoes = int(percentual * len(Y))
        indices_inversao = np.random.choice(len(Y), num_inversoes, replace=False)
        Y[indices_inversao] *= -1
        return Y
    
    def calcular_erro(self, X, Y, w):
        predicoes = np.sign(np.dot(X, w))
        return np.mean(predicoes != Y)

    def experimento4(self, N1, N2, i, pesos_iniciais='zeros'):
        E_ins = []
        E_outs = []
        
        for _ in range(1000):
            pesos_fx = np.zeros(2)  # Inicializando os pesos fx como zeros
            X1, Y1 = self.criar_pontos_e_rotulos(N1, pesos_fx)
            X2, Y2 = self.criar_pontos_e_rotulos(N2, pesos_fx)  # Usando os mesmos pesos_fx para N2
            Y1 = self.inverter_rotulos(Y1, percentual=0.1)
            Y2_original = np.copy(Y2)
            
            if pesos_iniciais == 'regressao_linear':
                reg = LinearRegression()
                pesos = reg.criar_pesos_fx(np.vstack((X1, X2)))
            else:
                pesos = np.zeros(3)
                
            pla = LinearRegression()
            w_pocket = pla.pocket_pla(np.hstack((np.ones((N1, 1)), X1)), Y1, max_iter=i)
            E_in = pla.calcular_erro(np.hstack((np.ones((N1, 1)), X1)), Y1, w_pocket)
            E_out = pla.calcular_erro(np.hstack((np.ones((N2, 1)), X2)), Y2_original, w_pocket)
            
            E_ins.append(E_in)
            E_outs.append(E_out)
        
        return np.mean(E_ins), np.mean(E_outs)




# Instanciar a classe
lin_reg = LinearRegression()

# Executar regressão linear e plotar resultados
pesos_fx, pesos_gx, X, Y, w_gx_novo, X_ext, Y_ext, E_in = lin_reg.regressao_linear()
lin_reg.plotar_pontos_e_linhas(pesos_fx, pesos_gx, X, Y, E_in)

# Realizar experimento e calcular E_in médio
w_fs, w_gs, E_in_medio = lin_reg.experimento()
print("O E_in médio após 1000 interações, para N= 100 é: {}".format(E_in_medio))

# Calcular E_out médio
E_out_medio = lin_reg.experimento2(w_fs, w_gs)
print("Para 1000 interações com 1000 pontos novos, o E_out médio é = {}".format(round(E_out_medio, 3)))

# Calcular número médio de iterações para convergência do PLA
media_iteracoes_PLA = lin_reg.experimento3()
print("Número médio de iterações para convergência do PLA com N= 10 pontos usando w_gx da regressão linear como 'good start' em 1000 interações: {}".format(round(media_iteracoes_PLA)))

# (a) Inicializando os pesos com 0; i = 10; N1 = 100; N2 = 1000.
E_in_a, E_out_a = lin_reg.experimento4(N1=100, N2=1000, i=10, pesos_iniciais='zeros')
print("(a) E_in médio:", E_in_a)
print("(a) E_out médio:", E_out_a)

# (b) Inicializando os pesos com 0; i = 50; N1 = 100; N2 = 1000.
E_in_b, E_out_b = lin_reg.experimento4(N1=100, N2=1000, i=50, pesos_iniciais='zeros')
print("(b) E_in médio:", E_in_b)
print("(b) E_out médio:", E_out_b)

# (c) Inicializando os pesos com Regressão Linear; i = 10; N1 = 100; N2 = 1000.
E_in_c, E_out_c = lin_reg.experimento4(N1=100, N2=1000, i=10, pesos_iniciais='regressao_linear')
print("(c) E_in médio:", E_in_c)
print("(c) E_out médio:", E_out_c)

# (d) Inicializando os pesos com Regressão Linear; i = 50; N1 = 100; N2 = 1000.
E_in_d, E_out_d = lin_reg.experimento4(N1=100, N2=1000, i=50, pesos_iniciais='regressao_linear')
print("(d) E_in médio:", E_in_d)
print("(d) E_out médio:", E_out_d)