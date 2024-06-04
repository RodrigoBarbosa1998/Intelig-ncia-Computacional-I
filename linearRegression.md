# Regressão Linear

Nestes problemas, nós vamos explorar como Regressão Linear pode ser usada em tarefas de classificação. Você usará o mesmo esquema de produção de pontos visto na parte acima do Perceptron, com \(d = 2\), \(X = [-1, 1] \times [-1, 1]\), e assim por diante.

1. **Considere \(N = 100\)**. Use Regressão Linear para encontrar \(g\) e calcule \(E_{in}\), a fração de pontos dentro-de-amostra que foram classificados incorretamente (armazene os \(g\)'s pois eles serão usados no item seguinte). Repita o experimento 1000 vezes. Qual dos valores abaixo é mais próximo do \(E_{in}\) médio?
    - (a) 0
    - (b) 0.001
    - (X) 0.01
    - (d) 0.1
    - (e) 0.5

2. **Agora, gere 1000 pontos novos** e use eles para estimar o \(E_{out}\) dos \(g\)'s que você encontrou no item anterior. Novamente, realize 1000 execuções. Qual dos valores abaixo é mais próximo do \(E_{out}\) médio?
    - (a) 0
    - (b) 0.001
    - (X) 0.01
    - (d) 0.1
    - (e) 0.5

3. **Agora, considere \(N = 10\)**. Depois de encontrar os pesos usando Regressão Linear, use-os como um vetor de pesos iniciais para o Algoritmo de Aprendizagem Perceptron (PLA). Execute o PLA até que ele convirja num vetor final de pesos que separa perfeitamente os pontos dentro-de-amostra. Dentre as opções abaixo, qual é mais próxima do número médio de iterações (sobre 1000 execuções) que o PLA demora para convergir?
    - (a) 1
    - (X) 15
    - (c) 300
    - (d) 5000
    - (e) 10000

4. **Vamos agora avaliar o desempenho da versão pocket do PLA em um conjunto de dados que não é linearmente separável**. Para criar este conjunto, gere uma base de treinamento com \(N_1\) pontos como foi feito até agora, mas selecione aleatoriamente 10% dos pontos e inverta seus rótulos. Em seguida, implemente a versão pocket do PLA, treine-a neste conjunto não-linearmente separável, e avalie seu \(E_{out}\) numa nova base de \(N_1\) pontos na qual você não aplicará nenhuma inversão de rótulos. Repita para 1000 execuções, e mostre o \(E_{in}\) e \(E_{out}\) médios para as seguintes configurações (não esqueça dos gráficos scatterplot, como anteriormente):

    - (a) Inicializando os pesos com 0; \(i = 10\); \(N_1 = 100\); \(N_2 = 1000\).
        - E_in médio: 0.19576
        - E_out médio: 0.12951
    - (b) Inicializando os pesos com 0; \(i = 50\); \(N_1 = 100\); \(N_2 = 1000\).
        - E_in médio: 0.11723
        - E_out médio: 0.03051
    - (c) Inicializando os pesos com Regressão Linear; \(i = 10\); \(N_1 = 100\); \(N_2 = 1000\).
        - E_in médio: 0.19499
        - E_out médio: 0.12964
    - (d) Inicializando os pesos com Regressão Linear; \(i = 50\); \(N_1 = 100\); \(N_2 = 1000\).
        - E_in médio: 0.11815
        - E_out médio: 0.03208