### Regressão Não-Linear

Nestes problemas, nós vamos novamente aplicar Regressão Linear para classificação. Considere a função target: 

\[ f(x_1, x_2) = \text{sign}(x_1^2 + x_2^2 - 0.6) \]

Gere um conjunto de treinamento de \( N = 1000 \) pontos em \( X = [-1, 1] \times [-1, 1] \) com probabilidade uniforme escolhendo cada \( x \in X \). Gere um ruído simulado selecionando aleatoriamente 10% do conjunto de treinamento e invertendo o rótulo dos pontos selecionados.

#### 1. Execute a Regressão Linear sem nenhuma transformação, usando o vetor de atributos (1, x2, x2) para encontrar o peso w. Qual é o valor aproximado de classificação do erro médio dentro-de-amostra \( E_{in} \) (medido ao longo de 1000 execuções)?

(a) 0
(b) 0.1
(c) 0.3
(X) 0.5
(e) 0.8

#### 2. Agora, transforme os \( N = 1000 \) dados de treinamento seguindo o vetor de atributos não-linear (1, x2, x2, x1x2, x2^2, x2^2). Encontre o vetor \( w_e \) que corresponda à solução da Regressão Linear. Quais das hipóteses a seguir é a mais próxima à que você encontrou? Avalie o resultado médio obtido após 1000 execuções.

(X) \( g(x_1, x_2) = \text{sign}(-1 - 0.05x_1 + 0.08x_2 + 0.13x_1x_2 + 1.5x_2^2 + 1.5x_2^2) \)
(b) \( g(x_1, x_2) = \text{sign}(-1 - 0.05x_1 + 0.08x_2 + 0.13x_1x_2 + 1.5x_2^2 + 15x_2^2) \)
(c) \( g(x_1, x_2) = \text{sign}(-1 - 0.05x_1 + 0.08x_2 + 0.13x_1x_2 + 15x_2^2 + 1.5x_2^2) \)
(d) \( g(x_1, x_2) = \text{sign}(-1 - 1.5x_1 + 0.08x_2 + 0.13x_1x_2 + 0.05x_2^2 + 0.05x_2^2) \)
(e) \( g(x_1, x_2) = \text{sign}(-1 - 0.05x_1 + 0.08x_2 + 1.5x_1x_2 + 0.15x_2^2 + 0.15x_2^2) \)

#### 3. Qual o valor mais próximo do erro de classificação fora-de-amostra \( E_{out} \) de sua hipótese na questão anterior? (Estime-o gerando um novo conjunto de 1000 pontos e usando 1000 execuções diferentes, como antes).

(a) 0
(X) 0.1
(c) 0.3
(d) 0.5
(e) 0.8
