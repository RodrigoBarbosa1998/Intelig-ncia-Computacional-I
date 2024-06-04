import numpy as np
from numpy.linalg import inv

class NonlinearRegression:
    
    def __init__(self, num_points=1000):
        self.num_points = num_points
    
    def gerar_pontos_e_rotulos(self, N):
        """Gera N pontos aleatórios no espaço [-1, 1] x [-1, 1] e rotula-os com base na nova função target.
        Args:
            N: Número de pontos aleatórios
        Retorna:
            Uma tupla contendo:
                X: um array N x D representando as coordenadas dos pontos
                Y: um array N x 1 representando +1 ou -1 dependendo do lado da função target 
        """
        X = np.random.uniform(-1, 1, (N, 2))
        Y = X[:, 0]**2 + X[:, 1]**2 - 0.6 >= 0
        Y = np.where(Y, 1, -1)
        random_idx = np.random.choice(np.arange(len(Y)), N // 10, replace=False) # adiciona ruído a 10% dos pontos
        Y[random_idx] = -Y[random_idx]
        return X, Y

    def regressao_linear_sem_transformacao(self, N):
        X, Y = self.gerar_pontos_e_rotulos(N)
        
        X_transformado = np.hstack((np.ones((N, 1)), X))
        Y_transformado = np.expand_dims(Y, 1)
        
        w_pseudo = np.dot(inv(np.dot(X_transformado.T, X_transformado)), X_transformado.T)
        w_gx = np.dot(w_pseudo, Y_transformado)
        
        Y_pred = np.where(np.dot(X_transformado, w_gx) > 0, 1, -1)
        E_in = np.mean(Y_pred != np.expand_dims(Y, 1))
        return E_in

    def experimento_sem_transformacao(self, tentativas=1000, N=1000):
        E_ins = np.zeros(tentativas)
        for i in range(tentativas):
            E_in = self.regressao_linear_sem_transformacao(N)
            E_ins[i] = E_in
        return np.mean(E_ins)

    def regressao_linear_com_transformacao(self, N):
        X, Y = self.gerar_pontos_e_rotulos(N)
        
        X_transformado = np.hstack((
            np.ones((N, 1)),
            X,
            (X[:, 0] * X[:, 1])[:, np.newaxis],
            (X[:, 0]**2)[:, np.newaxis],
            (X[:, 1]**2)[:, np.newaxis]
        ))
        
        Y_transformado = np.expand_dims(Y, 1)
        
        w_pseudo = np.dot(inv(np.dot(X_transformado.T, X_transformado)), X_transformado.T)
        w_gx = np.dot(w_pseudo, Y_transformado)
        
        Y_pred = np.where(np.dot(X_transformado, w_gx) > 0, 1, -1)
        E_out = np.mean(Y_pred != np.expand_dims(Y, 1))
        return E_out, w_gx

    def experimento_com_transformacao(self, tentativas=1000, N=1000):
        E_outs = np.zeros(tentativas)
        for i in range(tentativas):
            E_out = self.regressao_linear_com_transformacao(N)[0]
            E_outs[i] = E_out
        return np.mean(E_outs)


# Instancia a classe NonlinearRegression
regressor = NonlinearRegression()

# Executa o experimento sem transformação e exibe o erro médio E_in
E_in_medio = regressor.experimento_sem_transformacao()
print("Para 1000 interações com 1000 pontos, o E_in médio é = {}".format(round(E_in_medio, 1)))

# Executa a regressão linear com transformação e exibe os pesos
w_tilde = regressor.regressao_linear_com_transformacao(1000)[1]
print("Os pesos w_tilde são aproximadamente {}".format(np.round(w_tilde.T, 2)))

# Executa o experimento com transformação e exibe o erro médio E_out
E_out_medio = regressor.experimento_com_transformacao()
print("Para 1000 interações com 1000 pontos, o E_out médio é = {}".format(round(E_out_medio, 1)))
