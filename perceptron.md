# Perceptron

Neste problema, você criará a sua própria função target \( f \) e uma base de dados \( D \) para que possa ver como o Algoritmo de Aprendizagem Perceptron funciona. Escolha \( d = 2 \) para que você possa visualizar o problema, e assuma \( X = [-1, 1] \times [-1, 1] \) com probabilidade uniforme de escolher cada \( x \in X \).

Em cada execução, escolha uma reta aleatória no plano como sua função target \( f \) (faça isso selecionando dois pontos aleatórios, uniformemente distribuídos em \([-1, 1] \times [-1, 1]\), e pegando a reta que passa entre eles), de modo que um lado da reta mapeia para +1 e o outro para -1. Escolha os inputs \( x_n \) da base de dados como um conjunto de pontos aleatórios (uniformemente em \( X \)), e avalie a função target em cada \( x_n \) para pegar o output correspondente \( y_n \).

Agora, para cada execução, use o Algoritmo de Aprendizagem Perceptron (PLA) para encontrar \( g \). Inicie o PLA com um vetor de pesos \( w \) zerado (considere que sign(0) = 0, de modo que todos os pontos estejam classificados erroneamente ao início), e a cada iteração faça com que o algoritmo escolha um ponto aleatório dentre os classificados erroneamente. Estamos interessados em duas quantidades: o número de iterações que o PLA demora para convergir para \( g \), e a divergência entre \( f \) e \( g \) que é \( P[f(x) \ne g(x)] \) (a probabilidade de que \( f \) e \( g \) vão divergir na classificação de um ponto aleatório). Você pode calcular essa probabilidade de maneira exata, ou então aproximá-la ao gerar uma quantidade suficientemente grande de novos pontos para estimá-la (por exemplo, 10.000).

A fim de obter uma estimativa confiável para essas duas quantias, você deverá realizar 1000 execuções do experimento (cada execução do jeito descrito acima), tomando a média destas execuções como seu resultado final.

Para ilustrar os resultados obtidos nos seus experimentos, acrescente ao seu relatório gráficos scatterplot com os pontos utilizados para calcular \( E_{out} \), assim como as retas correspondentes à função target e à hipótese \( g \) encontrada.

1. Considere \( N = 10 \). Quantas iterações demora, em média, para que o PLA convirja com \( N = 10 \) pontos de treinamento? Escolha o valor mais próximo do seu resultado.
   1. (a) 1
   2. (X) 15
   3. (c) 300
   4. (d) 5000
   5. (e) 10000

2. Qual das alternativas seguintes é mais próxima de \( P[f(x) \ne g(x)] \) para \( N = 10 \)?
   1. (a) 0.001
   2. (X) 0.01
   3. (c) 0.1
   4. (d) 0.5
   5. (e) 1

3. Agora considere \( N = 100 \). Quantas iterações demora, em média, para que o PLA convirja com \( N = 100 \) pontos de treinamento? Escolha o valor mais próximo do seu resultado.
   1. (a) 50
   2. (X) 100
   3. (c) 500
   4. (d) 1000
   5. (e) 5000

4. Qual das alternativas seguintes é mais próxima de \( P[f(x) \ne g(x)] \) para \( N = 100 \)?
   1. (X) 0.001
   2. (b) 0.01
   3. (c) 0.1
   4. (d) 0.5
   5. (e) 1

## Resposta prompt (1 - 4):
### Executando para conjunto de dados N=10, Iterações= 1000 vezes, e avaliando cada hipótese contra um conjunto de teste de 10.000 pontos.
Média de iterações para convergir para o conjunto de dados N=10: 17.331
Média P[f(x) != g(x)]: 0.0089

### Executando para conjunto de dados N=100, Iterações= 1000 vezes, e avaliando cada hipótese contra um conjunto de teste de 10.000 pontos.
Média de iterações para convergir para o conjunto de dados N=100: 100.389
Média P[f(x) != g(x)]: 0.0009   

5. É possível estabelecer alguma regra para a relação entre \( N \), o número de iterações até a convergência, e \( P[f(x) \ne g(x)] \)?
    •À medida que N aumenta, o número médio de iterações até a convergência também aumenta. No entanto, P[f(x)≠g(x)] tende a diminuir, indicando que a hipótese g se aproxima mais da função target f com um conjunto de dados maior.
    •Essa relação sugere que mais pontos de treinamento resultam em uma hipótese mais precisa, mas requerem mais iterações para o PLA convergir.
    •Sim, é possível observar que, à medida que N aumenta, o número de iterações até a convergência tende a aumentar, enquanto P[f(x)≠g(x)] tende a diminuir, indicando uma hipótese mais precisa.
