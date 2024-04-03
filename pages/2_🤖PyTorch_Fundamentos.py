import streamlit as st
import torch
import pytorch_lightning as pl
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd


PAGE_TITLE = 'PyTorch Fundamentos 🤖'
PAGE_ICON = "🤖"
st.set_page_config(page_title=PAGE_TITLE,page_icon=PAGE_ICON)

# Menu lateral
with st.sidebar:
    st.image('imagens/pytorch-logo.png',width=200)
    st.sidebar.title('PyTorch Fundamentos')
    selected = option_menu("",['0 - O que são tensores?',
                               '1 - Criando Tensores',
                               '2 - Trabalhando com as dimensões dos Tensores',
                               '3 - Operações aritméticas com Tensores',
                               '4 - Concatenação, Expansão, Junção, Chunk, Squeeze'],
                               icons=['bi bi-droplet',
                                      'bi bi-droplet',
                                      'bi bi-droplet',
                                      'bi bi-droplet',
                                      'bi bi-droplet'],
                               default_index=0)

st.write("[Conheça o meu GitHub](https://github.com/AurelioGuilherme)")
st.write("[Documentação PyTorch](https://pytorch.org/docs/stable/index.html)")


def main():
    if selected == '0 - O que são tensores?':
        st.write('# **O que são tensores?**')
        st.write("""
                    O objeto tensor utilizado no framework PyTorch é focado 
                    e projetado para processamento paralelizado. 
        
                    Ele representa uma estrutura de dados multidimensional 
                    que pode ser manipulada de forma eficiente em GPUs e CPUs. 
        
                    Essa capacidade de processamento paralelo é essencial 
                    para lidar com grandes conjuntos de dados e realizar 
                    cálculos complexos de forma eficiente. 
        
                    Os tensores no PyTorch são a base para a construção de 
                    modelos de aprendizado profundo e outras tarefas de 
                    computação científica, fornecendo uma maneira flexível 
                    e eficaz de representar e manipular dados em várias dimensões.
                 """)

        st.image('imagens/tensor.png')

        st.write("""
                    Os tensores são amplamente utilizados em várias áreas, 
                    incluindo aprendizado de máquina, visão computacional, 
                    processamento de linguagem natural e física, entre outros. 
        
                    Eles fornecem uma maneira flexível e eficiente de 
                    representar e manipular dados em várias dimensões. 
        
                    No aprendizado de máquina, os tensores são usados para 
                    representar conjuntos de dados, parâmetros de modelo, 
                    gradientes durante o treinamento e resultados de predição. 
        
                    Eles são a base para a construção de modelos de aprendizado 
                    profundo, como redes neurais convolucionais e redes neurais recorrentes. 
        
                    Além disso, em computação científica, os tensores são 
                    usados para representar tensores de tensão em mecânica, 
                    campos vetoriais em física e muito mais. 
                    Sua capacidade de processamento paralelo em hardware 
                    especializado os torna essenciais para lidar com grandes 
                    volumes de dados e realizar cálculos complexos de forma eficiente.
                """)
        st.image('imagens/img-1.png')

        st.write("""
                    Os tensores têm uma ampla gama de aplicações práticas em diversas áreas. 
                 
                    No processamento de imagens, os tensores são usados para representar 
                    imagens digitais em três ou mais dimensões (largura, altura e canais de cor). 
        
                    Isso permite realizar operações como convoluções e pooling em imagens 
                    para tarefas como classificação e detecção de objetos. 
        
                    Em processamento de linguagem natural, os tensores são usados para 
                    representar sequências de palavras em texto e realizar operações 
                    como embeddings e atenção em modelos de processamento de linguagem. 
        
                    Além disso, em física e engenharia, os tensores são usados 
                    para representar grandezas físicas como tensão, deformação e 
                    fluxo de calor em sistemas complexos.
                """)
        st.image('imagens/April-28-deep-learning-applications-infograph.png')
        
    elif selected == '1 - Criando Tensores':
        st.write('# **Criando Tensores**')
        st.write("### Tensores PyTorch x Arrays NumPy:\n")

        st.write("""
                    Os tensores no PyTorch são estruturas de dados multidimensionais 
                    que podem ter uma ou mais dimensões. As dimensões de um tensor 
                    representam a forma ou o tamanho de cada eixo do tensor. Por exemplo, 
                    um tensor bidimensional tem duas dimensões: uma dimensão 
                    para linhas e outra para colunas. 
                """)

        st.write('Você pode criar tensores a partir de listas ou matrizes numpy e vice-versa utilizando a função `.Tensor()`')
        st.write('Esta função cria um objeto do tipo tensor')
        lista_python = [[1,2,3], [4,5,6]]
        t1 = torch.Tensor(lista_python)  

        # Mostrar o código
        with st.expander("**Cria um tensor 2x3 a partir de uma lista Python**"):
            st.code("""
                        import torch

                        # Cria um tensor 2x3 a partir de uma lista Python
                        lista_python = [[1,2,3], [4,5,6]]
                        t1 = torch.Tensor(lista_python)
                        print(t1)
                    """, language="python")
            st.write('**Valor do tensor t1 criado com uma lista python**')
            st.write(t1)

        array_numpy = np.array([[9,6,1], [5,3,2]])
        t2 = torch.Tensor(array_numpy)

        # Mostrar código
        with st.expander('**Cria um tensor 2x3 a partir de array numpy**'):
            st.code('''
                        import torch
                        import numpy as np

                        # Cria um tensor 2x3 a partir de um array Numpy                    
                        array_numpy = np.array([[9,6,1], [5,3,2]])
                        t2 = torch.Tensor(array_numpy)
                        print(t2)                    
                    ''', language='python')
            st.write('**Valor do tensor t2 criado com uma array numpy**')
            st.write(t2)

        with st.expander('**Criando um tensor com range de valores com o método `.arange`**'):
            st.write('É possivel criar um Tensor com um range de valores com o `.arange`')
            v = torch.arange(5)
            st.code('''
                       # Criando tensores com range de valores
                       v = torch.arange(5)
                       print(v)
                   ''',language='python')
            st.write('**Valor do Tensor v de 1 dimensão**')
            st.write(v)
            st.write('''O tensor criado é em tensor em 1 dimensão, 
                        para mudar a dimensão podemos utilizar os metodos`.reshape()` e o `.view()` .''')
            st.code('''
                       #Criando tensor em 1 dimensão com arange
                       v = torch.arange(9)
                       print(v)
                   ''',language='python')
            v = torch.arange(9)
            st.write('**Valor tensor v em 1 dimensão**')
            st.write(v)
            st.code('''
                        # Alterando o tensor para 2 dimensões
                        v = v.view(3,3)
                        print(v)
                    ''',language='python')
            v = v.view(3,3)
            st.write('**Valor do tensor v em 2 dimensões**')
            st.write(v)
        with st.expander('**Criando um tensor Linear com `.linspace()`**'):
            st.write('''
                       O `torch.linspace()` é particularmente útil
                       para gerar tensores lineares com valores igualmente espaçados ao 
                       longo de um intervalo especificado.

                        Há três argumentos principais: o valor inicial do intervalo,
                        o valor final do intervalo e o número de elementos desejados no tensor 
                     ''')
            v = torch.linspace(1, 10, steps=10)
            st.code(''' 
                       # Cria um tensor com 10 pontos lineares de (1, 10)
                       v = torch.linspace(1, 10, steps = 10)

                     ''',language='python')
            st.write('**Valor do tensor linear v**')
            st.write(v)

        with st.expander('**Criando um tensor em escala logarítimica**'):
            st.write('''
                        Ao usar `torch.logspace()`, fornecemos três argumentos principais: 
                        o expoente inicial, o expoente final e o número de elementos desejados no 
                        tensor. O PyTorch então retorna um tensor com valores distribuídos de forma 
                        logarítmica entre 10^inicio e 10^fim, inclusive.
                   ''')
            v = torch.logspace(start = -1, end = 10, steps = 5 )
            st.code(''' 
                        # Criando tensor em escala logarítimica    
                        v = torch.logspace(start = -1, end = 10, steps = 5 )

                    ''',language='python')
            st.write('**Valor do tensor logarítimico**') 
            st.write(v)

        with st.expander('**Criando um tensor com valores absolutos**'):
            st.write('''
                       Para criar um tensor com valores absolutos em PyTorch,
                       você pode usar a função `torch.abs()` para calcular os valores 
                       absolutos de um tensor existente ou pode criar um tensor com 
                       valores absolutos diretamente.''')
            f = torch.FloatTensor([-1, -2, 3])
            r = torch.abs(f)
            st.code('''
                        # Criando um tensor com valores negátivos    
                        f = torch.FloatTensor([-1, -2, 3])
                    
                        # Convertendo o tensor para valores absolutos.
                        r = torch.abs(f)
                        print(r)
                    ''',language='python')
            st.write('**Valor do tensor r com valores absolutos**')
            st.write(r)
        

        st.write('### Tipos de Tensores no PyTorch')
        st.write('''
                    Ao criar tensores, podemos especificar suas dimensões e 
                    inicializá-los com diferentes valores. Por exemplo, 
                    podemos criar um tensor de zeros, onde todos os elementos 
                    do tensor têm o valor zero. Isso é útil para inicializar 
                    tensores antes de realizar operações ou preenchê-los com 
                    dados reais posteriormente. Podemos criar um tensor de zeros 
                    usando a função `torch.zeros(dimensões)`, onde "dimensões" é 
                    uma lista ou tupla que especifica o tamanho de cada dimensão 
                    do tensor.

                    No entanto, é importante ter cuidado ao trabalhar com tensores 
                    e gerenciar a memória corretamente. Às vezes, ao criar ou manipular 
                    tensores, podemos gerar `"lixo de memória"`, que são áreas de memória 
                    alocadas para objetos que não estão mais em uso, mas ainda não foram liberadas. 
                    Isso pode levar a vazamentos de memória e redução do desempenho do programa. 
                    Para evitar o lixo de memória, é importante liberar os recursos adequadamente 
                    após o uso, usando métodos como "del" em tensores ou utilizando o mecanismo 
                    de coleta de lixo do Python.
                ''')
        a = torch.rand(2,2,3)

        # Mostrar código
        with st.expander('Cria um tensor 2 x 2 x 3 com valores randômicos'):
            st.code('''
                        import torch

                        # Cria um tensor 2 x 2 x 3 com valores randômicos                   
                        a = torch.rand(2,2,3)
                        print(a)                    
                    ''', language='python')
            st.write(a)

        b = torch.zeros(2,2,3)

        # Mostrar código
        with st.expander('Criando um tensor preenchido com zeros'):
            st.code('''
                        import torch

                        # Criando um tensor preenchido com zeros                  
                        b = torch.zeros(2,2,3)
                        print(b)                    
                    ''', language='python')
            st.write(b)

        c = torch.zeros_like(a)
        
        # Mostrar código
        with st.expander('Criando um tensor semelhante a outro'):
            st.code('''
                        import torch
                                                                 
                        # Criando um tensor semelhante a outro              
                        c = torch.zeros_like(a)
                        print(c)                    
                    ''', language='python')
            st.write(c)

        d = torch.ones_like(a)

        # Mostrar código
        with st.expander("**Criando um tensor de 1's semelhante (like) ao tensor de zeros (mesmas dimensões) `torch.ones_like()`**"):
            st.code('''
                        import torch

                        # Criando um tensor de 1's semelhante (like) ao tensor de zeros (mesmas dimensões)             
                        d = torch.ones_like(a)
                        print(d)                    
                    ''', language='python')
            st.write(d, unsafe_allow_html=True)
        
        with st.expander('**Tensor Diagonal `torch.diag()`**'):
             st.write('''Uma das operações comuns em álgebra linear é a criação de uma matriz diagonal, 
                      na qual todos os elementos fora da diagonal principal são zeros e os elementos 
                      na diagonal principal são iguais. Em PyTorch, podemos facilmente criar um tensor 
                      diagonal de uns utilizando a função `torch.diag()`.''')
             st.code('''
                        # Criando tensores de 1's  
                        v = torch.ones(3)
                        print(v)

                        # Transpondo para um Tensor de Size 3x3
                        r = torch.diag(v)
                        print(r)
                ''',language='python')
             v = torch.ones(3)
             st.write("**Tensor de 1's**")
             st.write(v)
             r = torch.diag(v)
             st.write('**Vetor Diagonal transposto**')
             st.write(r.numpy())
             



        st.write('### Criando tensores de diferentes tipos')
        st.write("""
        No PyTorch, existem diferentes tipos de tensores, cada um com suas próprias características e finalidades específicas. Os principais tipos de tensores são:

        1. Tensor Float: Este tipo de tensor é utilizado para representar números reais, sendo frequentemente empregado em tarefas de aprendizado de máquina, onde a precisão decimal é importante. Podemos encontrar tensores float de 32 bits (torch.float32 ou torch.float) e 64 bits (torch.float64 ou torch.double), sendo que o primeiro é mais comumente utilizado devido à sua eficiência computacional.

        2. Tensor Inteiro: Este tipo de tensor é utilizado para representar números inteiros. Assim como os tensores float, podemos encontrar tensores inteiros de diferentes tamanhos, como torch.int8, torch.int16, torch.int32 e torch.int64, dependendo da precisão necessária para a aplicação específica.

        3. Tensor Booleano: Tensores booleanos são utilizados para representar valores lógicos, ou seja, verdadeiro ou falso. Eles são frequentemente utilizados em operações de máscara e indexação.

        4. Tensor Byte: Este tipo de tensor é semelhante ao tensor booleano, mas pode armazenar valores inteiros de 0 a 255, ocupando menos espaço de memória do que um tensor inteiro de 32 bits. É comumente utilizado em operações de processamento de imagens.

        A existência de múltiplos tipos de tensores no PyTorch se deve à necessidade de flexibilidade e eficiência em diferentes cenários de aplicação. Cada tipo de tensor oferece um compromisso entre precisão e eficiência computacional, permitindo aos desenvolvedores escolher o tipo mais adequado para a sua aplicação específica. Isso permite otimizar o desempenho e o consumo de recursos do modelo, garantindo ao mesmo tempo a precisão necessária para as tarefas em questão.
                
        **Vejamos um exemplo de criação tensores:**""")

        a = np.array([[4,5,6], [7,8,9]])
        with st.expander('Criando numpy array'):
            st.code('''
                        import numpy as np

                        # Array numpy           
                        a = np.array([[4,5,6], [7,8,9]])
                        print(f'Tipo de dado Array Numpy: {a.dtype}')
                                       
                    ''', language='python')
            st.write(f'Tipo de dado Array Numpy: {a.dtype}')
            st.write(a)

        st.write('Por padrão o PyTorch utiliza o `FloatTensor` ao criar um objeto da classe tensor.')
        st.write('### Ponto de atenção:')
        st.write('**CUIDADO:** Um objeto Tensor não inicializado contém dados de lixo de memória!')
        
        b = torch.Tensor(a)
        with st.expander('Criando tensor utilizando o array'):
            st.code('''
                        import torch
                    
                        # Criando um Tensor.       
                        b = torch.Tensor(a)
                        print(f'Tipo de dado `Tensor`: {b.type()}')
                                       
                    ''', language='python')
            st.write(f"Tipo de dado Tensor: {b.type()}", unsafe_allow_html=True)
            st.write(b)
        
        c = torch.FloatTensor(a)
        with st.expander('Criando tensor um `FloatTensor` utilizando um array.'):
            st.code('''
                        import torch
                    
                        # Criando um FloatTensor        
                        c = torch.FloatTensor(a)
                        print(f'Tipo de dado `FloatTensor`: {c.type()}')
                                       
                    ''', language='python')
            st.write(f"Tipo de dado Tensor: {c.type()}", unsafe_allow_html=True)
            st.write(c)

        d = torch.LongTensor(a)
        with st.expander('Criando tensor um `LongTensor` utilizando o array.'):
            st.code('''
                        import torch
                    
                        # Criando um LongTensor        
                        d = torch.LongTensor(a)
                        print(f'Tipo de dado Tensor: {d.type()}')
                                       
                    ''', language='python')
            st.write(f"Tipo de dado Tensor: {d.type()}", unsafe_allow_html=True)
            st.write(d)            

        e = [True, False,True, True, True, False]
        f = torch.Tensor(e)
        with st.expander('Criando tensor a partir da lista de booleanos'):
            st.code('''
                        import torch

                        # Criando tensor a partir da lista de booleanos          
                        e = [True, False, True, True, False]
                        f = torch.Tensor(e)
                        print(f)
                        print(f.type())           
                    ''', language='python')
            st.write(f)
            st.write(f.type())
        
        g = torch.zeros(10, dtype = torch.bool)
        with st.expander('Criando tensor com valores booleanos'):
            st.code('''
                        import torch

                        # Criando tensor com valores booleanos         
                        g = torch.zeros(10, dtype = torch.bool)
                        print(g)
                        print(g.type())           
                    ''', language='python')
            st.write(g)
            st.write(g.type())
        
        st.write('### Alterando o tipo do tensor:')
        st.write("""
        É comum a necessidade de converter tensores de um tipo para outro. Isso pode ser útil em várias situações, como quando precisamos garantir a consistência dos tipos de dados em operações matemáticas ou quando queremos adaptar os tensores para diferentes operações ou modelos.

        Para converter um tensor para outro tipo, o PyTorch oferece diversos métodos. Por exemplo, podemos utilizar o método `.float()` para converter um tensor para ponto flutuante, ou `.long()` para converter para números inteiros. Também é possível utilizar métodos como `.double()`, `.half()`, `.bool()`, entre outros, dependendo das necessidades específicas da aplicação.

        Vejamos um exemplo de conversão de tipos de tensores:
                 """)
        a = np.array([[4,5,6], [7,8,9]])
        c = torch.FloatTensor(a)
        valor = c
        c = c.long()
        
        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
                        import torch

                        # Alterando o tipo do tensor:
                        a = np.array([[4,5,6], [7,8,9]])
                        c = torch.FloatTensor(a)
                        print(c.type())
                        c = c.long()
                        print(c)
                        print(c.type())                                     
                    ''', language='python')
            st.write(valor.type())
            st.write(c)
            st.write(c.type())
            
    elif selected == '2 - Trabalhando com as dimensões dos Tensores':
        st.write('# **Trabalhando com as dimensões dos Tensores**')
        st.write('O slicing permite extrair partes específicas de um tensor, permitindo o acesso aos elementos desejados.')
        st.write('### Size e Shape de Tensores')
        st.write('Visualizando as dimensões dos tensores com `shape` e `size()`')
        st.write('`shape`: é um atributo')
        st.write('`size()`:  é um método')
        
        # Cria um tensor com valores randômicos
        torch.manual_seed(777)
        x = torch.randint(0, 10, size = (2, 3, 4))


        with st.expander('Mostrar código'):
            st.code('''
                        import torch

                        # Cria um tensor com valores randômicos
                        torch.manual_seed(777)
                        x = torch.randint(0, 10, size = (2, 3, 4))                   
                        print(x)

                        # Shape - Atributo do objeto tensor
                        x.shape

                        # Size - Método do objeto tensor
                        x.size()                    

                        # Número total de elementos no Tensor                                   
                        torch.numel(x)

                        # Alterando o size do tensor (mas sem mudar o tensor original)
                        print(x.view(2, 2, 6))                                   
                    ''', language='python')
            st.write(x)
            st.write(x.shape)
            st.write(x.size())
            st.write(torch.numel(x))


        st.write('### View')
        st.write('Alterando o size do tensor (mas sem mudar o tensor original) com `view()`')
        
        # Mostrar código
        with st.expander('Mostrar código'):
            st.code('''
                       import torch
                                                                
                       # Cria um tensor com valores randômicos
                       torch.manual_seed(777)
                       x = torch.randint(0, 10, size = (2, 3, 4))                   
                       
                       # Alterando o size do tensor (mas sem mudar o tensor original)
                       print(f"Altera de '{x.size()}' para '{x.view(2, 2, 6).size()}'\n")     
                       print(x.view(2, 2, 6))                                                     
                    ''', language='python')
            st.write(f"Altera de '{x.size()}' para '{x.view(2, 2, 6).size()}'\n")
            st.write(x.view(2, 2, 6))

        st.write('Também podemos usar o método `view()` para criar um tensor')
        t = torch.arange(60).view(3, 4, 5)
        with st.expander('Mostrar código:'):
            st.code('''
                        import torch

                        t = torch.arange(60).view(3, 4, 5)
                        print(t)
                        print(t.shape)
                        print(t.size())
                        print(torch.numel(t))
                    ''', language= 'python')
            st.write(t) 
            st.write(t.shape) 
            st.write(t.size()) 
            st.write(torch.numel(t))


        
        def create_tensor(dimensions):
            try:
                return torch.randint(low=0, high=11, size=dimensions)
            except ValueError:
                st.error('Digite dimensões válidas para o tensor (números inteiros separados por vírgula).')

        st.write("### Slicing de Tensores")
        st.write('''
        Para realizar o slicing de um tensor, precisamos especificar os índices ou intervalos ao longo de cada dimensão do tensor. Podemos usar notações de intervalo para especificar o slicing de forma concisa e intuitiva.''')
        
        st.write('''
                    Sintaxe:

                    `tensor[tensor_position_start:tensor_position_end, tensor_dimension_start:tensor_dimension_end , tensor_value_start:tensor_value_end]`

                    Parâmetros:

                    - tensor_position_start
                    - tensor_position_end
                    - tensor_dimension_start
                    - tensor_dimension_stop
                    - tensor_value_start
                    - tensor_value_stop
                ''')
        st.write('''
                 Vamos exemplificar com um exemplo interativo onde você pode fornecer as dimensões em um imput logo a baixo:
                 
                Neste exemplo é criado um tensor com dimensões customizadas de acordo com o input e com seed definida para ter a possibilidade replicação.
                
                Os valores dos tensores são valore inteiros de 0 a 10.
                 ''')
        st.write('### Exemplo com input:')
        torch.manual_seed(222)
        dim_input = st.text_input("Digite as dimensões do tensor separadas por vírgula (ex: 3,4,5):")

        if dim_input:
            dimensions = [int(dim) for dim in dim_input.split(',') if dim.isdigit()]
            dimensions = [max(0, min(dim, 9999999)) for dim in dimensions]
            x = create_tensor(dimensions)
            st.write(x)
            st.write('**Fazendo slice**')
            st.write(f'O tensor possui: {x.ndim} dimensões') 
            slice_input = st.text_input("Digite os valores para o slice do tensor separados por vírgula (limitado à quantidade de dimensões):")
            try:
                if slice_input:
                    slices = [int(s) for s in slice_input.split(',') if s.isdigit()]
                    result = x[tuple(slices)]  # Convertendo para uma tupla para indexação
                    st.write("Resultado do slicing:", result)
                    with st.expander('Mostrar código: '):
                        st.code(f'''
                                # Cria um tensor com valores randômicos
                                torch.manual_seed(222)
                                x = torch.randint(0, 10, size = {dimensions})
                                print(x)

                                # Slicing do tensor:
                                ### ps o print esta exibindo ':' ao inves de virgulas em casos especificos:
                                ### exemplo correto: print(x[0:1, 0:1, :3])
                                print(x[{str(slice_input).replace(',', ':')}])           
                                ''',language='python')
            except IndexError:
                st.error("Índices de slice inválidos.")

        st.write('''Fatiar um Tensor com slicing baseado em indexação é útil, 
                     mas pode ser impraticável com tensores de muitas dimensões.
                     Para fatiar um tensor 4D no PyTorch, você pode usar o método 
                     `tensor.narrow()`. Este método permite especificar as dimensões 
                     ao longo das quais deseja fatiar o tensor e os índices inicial e 
                     final de cada dimensão.''')
        
        st.write('**3 Dimensões**')
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with st.expander('Cria um tensor '):
                        st.code(f'''
                                x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                                print(x)
                                ''',language='python')
                        st.write(x)
        y = torch.narrow(x, 0, 0, 2)
        with st.expander('A partir da dimensão de índice 0, retorne as dimensões de índice 0 a 2 '):
                        st.code(f'''
                                y = torch.narrow(x, 0, 0, 2)
                                print(y)
                                ''',language='python')        
                        st.write(y)

        st.write('**4 Dimensões**')
        torch.manual_seed(333)
        tensor_4d = torch.randn(4,3,5,7)
        with st.expander('Cria um tensor de 4 dimensões'):
             st.code('''
                        torch.manual_seed(333)
                        tensor_4d = torch.randn(4,3,5,7)
                     
                        #Quantidade de dimensões
                        tensor_4d.dim()
                        print(tensor_4d)''',language='python')
             st.write(tensor_4d.dim())
             st.write(tensor_4d)
        sliced_tensor_4d = tensor_4d.narrow(2, 0, 2)
        with st.expander('A partir da dimensão de índice 2, retorne as dimensões entre índices 0 e 2. '):
             st.code('''
                        sliced_tensor_4d = tensor_4d.narrow(2, 0, 2)
                        sliced_tensor_4d.shape
                        print(sliced_tensor_4d)
                    ''',language='python')
             st.write(sliced_tensor_4d.shape)
             st.write(sliced_tensor_4d)
        
        st.write('**5 Dimensões**')
        torch.manual_seed(222)
        tensor_5d = torch.randn(4, 3, 5, 7, 3)
        with st.expander('Cria um tensor de 5 dimensões'):
            st.code(''' 
                        tensor_5d = torch.randn(4, 3, 5, 7, 3)
                        tensor_5d.dim()
                        print(tensor_5d)
                    ''', language='python')
            st.write(tensor_5d.dim())
            st.write(tensor_5d)
        
        sliced_tensor_5d = tensor_5d.narrow(2, 0, 2).narrow(3, 0, 1)
        with st.expander('A partir da dimensão de índice 2, retorne as dimensões entre índices 0 e 2. '):
             st.code('''
                        # Faça isso em todas as posições e colocações.
                        # Depois disso, a partir da dimensão de índice 3, retorne as dimensões os índices 0 e 1 (ou seja, somente índice 0)
                        sliced_tensor_5d = tensor_5d.narrow(2, 0, 2).narrow(3, 0, 1)

                        print(sliced_tensor_5d)
                    ''', language='python')
             st.write(sliced_tensor_5d)
    if selected == '3 - Operações aritméticas com Tensores':
        st.write('# **Operações aritméticas com Tensores**', unsafe_allow_html=True)
        st.write('''
                    Operações aritméticas são essenciais para a manipulação e transformação de dados, 
                    desempenhando um papel crítico em todas as etapas do desenvolvimento de modelos.

                    **Soma e Subtração:**
                    As operações de soma e subtração são simples e frequentemente usadas em operações de 
                    ajuste de parâmetros, atualização de gradientes e normalização de dados. Por exemplo, 
                    ao treinar um modelo, essas operações podem ser usadas para calcular a diferença entre 
                    a saída prevista e o valor real (erro), que é essencial para ajustar os pesos do modelo.

                ''')
        # Cria 2 tensores
        x = torch.rand(2, 3) 
        y = torch.rand(2, 3)

        # Operação de soma
        z1 = x + y
        with st.expander('**Operação de soma**'):
             st.code('''
                        # Cria 2 tensores
                        x = torch.rand(2, 3) 
                        y = torch.rand(2, 3)

                        # Operação de soma
                        z1 = x + y
                        print(z1)
                    ''',language='python')
             st.write('**Valor de x**')
             st.write(x)
             st.write('**Valor de y**')
             st.write(y)
             st.write('**Soma x + y**')
             st.write(z1)

        with st.expander('**Operação de soma com função `.add`**'):
             z2 = torch.add(x, y)  
             st.code('''
                        # Cria 2 tensores
                        x = torch.rand(2, 3) 
                        y = torch.rand(2, 3)

                        # Operação soma com `.add`
                        z2 = torch.add(x, y)  
                        print(z2)
                    ''',language='python')
             st.write('**Valor de x**')
             st.write(x)
             st.write('**Valor de y**')
             st.write(y)
             st.write('**Soma x + y**')
             st.write(z2)

        with st.expander('**Somando valores a todos os elementos do tensor com `.add`**'):
            r = torch.add(x, 10)
            st.code('''
                        # Somando o valor 10 aos elementos do objeto Tensor.
                        r = torch.add(x, 10)
                        print(r)

                    ''', language='python')
            st.write('**Valor do tensor r após soma**')
            st.write(r)
        
        with st.expander('**Subtração com o método `.sub()`**'):
            st.write('''
                       As formas anteriores para a soma como "x + y" também funcionam para
                       o caso da subtração "x-y".
                       O método `.sub()` também é possivel utilizar de forma in-place e com parâmetro out.
                    ''')
            # Criando um tensor
            x = torch.tensor([1, 2, 3])
            
            # Subtraindo um valor escalar
            y = torch.sub(x, 1)
            st.code('''
                        # Subtraindo valores
                        x = torch.Tensor([1,2,3])
                        y = torch.sub(x, 1)

                    ''',language='python')
            st.write('**Resultado da subtração**')
            st.write(y)
            
            

        st.write('### O Parâmetro "out" em Operações de Tensores no PyTorch')  
        st.write('''
                    No PyTorch, o parâmetro `out` em operações de tensores 
                    oferece uma maneira flexível de controlar onde o resultado 
                    da operação será armazenado. Em muitas situações, podemos 
                    querer atribuir o resultado de uma operação a uma variável 
                    específica ou a um local de memória predefinido, e o 
                    parâmetro "out" nos permite fazer isso de forma eficiente.
                 ''')
        v1 = torch.Tensor(2, 3)
        with st.expander('**Aplicação do parêmetro `out`**'):        
            
            st.code('''
                        # Criando um tensor
                        v1 = torch.Tensor(2, 3)
                        print(v1)
                    ''',language='python')
            st.write('**Tensor v1**')
            st.write(v1)

            torch.add(x, y, out = v1)
            st.code('''
                        # Podemos atribuir o resultado da operação a uma variável. 
                        # Todos os métodos de operação possuem um parâmetro out para armazenar o resultado.
                        torch.add(x, y, out = v1)
                    ''',language='python')
            st.write('**Tensor v1 com soma utilizando parametro `out`**')
            st.write(v1)

        st.write('### Aplicações do parâmetro Out') 
        st.write('''
                    **Controle de Memória:**
                    Ao realizar operações em tensores, especialmente em modelos 
                    de aprendizado profundo com grandes conjuntos de dados, é 
                    crucial otimizar o uso de memória. O parâmetro "out" permite 
                    controlar explicitamente onde o resultado da operação será 
                    armazenado, evitando alocações de memória desnecessárias e 
                    reduzindo a sobrecarga do sistema.

                    **Reutilização de Memória:**
                    Uma das vantagens do uso do parâmetro "out" é a capacidade de 
                    reutilizar a memória alocada para tensores existentes. Em vez de 
                    alocar novos tensores para armazenar o resultado de uma operação, 
                    podemos especificar um tensor existente como destino para o resultado, 
                    economizando recursos de memória e melhorando o desempenho geral.

                    **Eficiência de Código:**
                    Usar o parâmetro "out" também pode resultar em código mais limpo 
                    e legível, pois evita a necessidade de atribuir o resultado da operação 
                    a uma variável separada. Isso pode simplificar o fluxo de trabalho de 
                    desenvolvimento e facilitar a manutenção do código ao longo do tempo.
                    ''')
        
        st.write('### Operações In-place')
        st.write(''' 
                    As operações in-place são aquelas que modificam diretamente o 
                    tensor existente, sem criar um novo tensor para armazenar o 
                    resultado. Isso é feito alterando os valores dos próprios 
                    elementos do tensor, em vez de alocar memória para um novo 
                    tensor. Como resultado, as operações in-place são mais eficientes 
                    em termos de uso de memória e tempo de execução.''')
        st.write('''**Sintaxe e Exemplos:**
                    A sintaxe para realizar uma operação in-place em PyTorch é adicionar
                    um sublinhado `_` ao final do nome da operação. Por exemplo, a operação
                    de adição in-place é representada pelo método `add_()`.
                 ''')
        
        
            
        with st.expander('**Aplicação de In-place operation**'):
             st.code('''
                        # In-place operation
                        # Mesmo que: x = x + y
                        x.add_(y)   
                    ''',language='python')
             st.write('**Mesmo que: x = x + y**')
             st.write(x.add_(y))

        st.write('### Somando valores pela indexação')
        st.write('Também é possivel efetuar operções pela indexação semelhante ao numpy') 
        x = torch.rand(2, 3)                     
        x[:, 0] = 0
        st.code('''
                    x[:, 1]                          
                    x[:, 0] = 0
                    print(x)
                ''',language='python')
        st.write(x)

        st.write('### Mais operações - Estatística')
        with st.expander('**Soma cumulativa `torch.cumsum()`**'):
            st.write('''
                        O método `.cumsum()` é a função que calcula a soma cumulativa ao longo de um eixo 
                        específico do tensor, adicionando os elementos do tensor sequencialmente. 
                        Isso é útil em uma variedade de cenários, incluindo processamento de sinais, 
                        análise de séries temporais e em algoritmos de otimização.
                     
                        Ao usar o método `cumsum()` em um tensor, podemos controlar o eixo ao longo do 
                        qual desejamos calcular a soma cumulativa. Por padrão, a soma cumulativa é 
                        realizada ao longo do primeiro eixo do tensor, mas podemos especificar o 
                        eixo desejado como um parâmetro.
                    ''')
            x = torch.Tensor([[1,2,3],
                              [4,5,6],
                              [7,8,9]])
            st.code(''' 
                        # Criando um tensor de 2 dimensões
                        x = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
                        print(x)
                    ''',language='python')
            st.write('**Valor do tensor x**')
            st.write(x)
            
            r = torch.cumsum(x, dim=0)
            st.code(''' 
                        # Soma acumulada por coluna
                        r = torch.cumsum(x, dim = 0)
                        print(r)
                    ''', language='python')
            st.write('**Soma acumulada por coluna**')
            st.write(r)

            r = torch.cumsum(x, dim=1)
            st.code(''' 
                        # Soma acumulada por linha
                        r = torch.cumsum(x, dim = 1)
                        print(r)
                    ''', language='python')
            st.write('**Soma acumulada por linha**')
            st.write(r)

        with st.expander('**Média `torch.mean()`**'):
             st.write('''
                        Ao utilizar o método `torch.mean()` em um tensor, o PyTorch calcula a 
                        média de todos os elementos do tensor ou ao longo de um eixo especificado. 
                        Se nenhum eixo for especificado, a média de todos os elementos do 
                        tensor é calculada. No entanto, se um eixo for fornecido, o cálculo da média 
                        será realizado ao longo desse eixo.
                     ''')
             r = torch.mean(x)
             st.code('''
                        import torch
                        # Cria 1 tensor de 2 dimensões
                        x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                     
                        # Média dos valores do Tensor
                        r = torch.mean(x)       
                        print(r)

                    ''',language='python')
             st.write('**Média dos valores do Tensor**')
             st.write(r)

             r = torch.mean(x, 0)
             st.code('''
                        # Média por coluna
                        r = torch.mean(x, 0) 
                        print(r)
                     ''',language='python')
             st.write('**Média dos valores do Tensor por coluna**')
             st.write(r)

             r = torch.mean(x, 1)
             st.code('''
                        # Média por linha
                        r = torch.mean(x, 1) 
                        print(r)
                     ''',language='python')
             st.write('**Média dos valores do Tensor por linha**')
             st.write(r)
        
        with st.expander('**Desvio padrão `torch.std()`**'):
             st.write('''
                        O desvio padrão é uma medida estatística que indica a dispersão 
                        dos valores em torno da média de um conjunto de dados. No PyTorch, 
                        o cálculo do desvio padrão de tensores é facilitado pelo método `.std()`,
                         que retorna o desvio padrão dos elementos do tensor.

                        Ao utilizar o método `.std()` em um tensor em PyTorch, podemos calcular 
                        rapidamente o desvio padrão de todos os elementos do tensor ou ao longo 
                        de um eixo específico em tensores multidimensionais. Se nenhum eixo for 
                        especificado, o desvio padrão será calculado para todos os elementos do tensor.
                     ''')
             st.code('''
                        import torch
                        # Cria 1 tensor de 2 dimensões
                        x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                       
                        # Desvio padrão do tensor
                        print(x.std())
                     
                        # Desvio padrão por linha
                        print(x.std(dim =1))
                    ''',language='python')
             st.write('**Desvio padrão do tensor**')
             st.write(x.std())
             st.write('**Desvio padrão do tensor por linha**')
             st.write(x.std(dim = 1))

        with st.expander('**Soma dos elementos do tensor `torch.sum()`**'):
            st.write('''
                        Ao usar o método .sum() em um tensor, podemos controlar o eixo ao 
                        longo do qual desejamos calcular a soma. Por padrão, a soma é realizada 
                        em todos os elementos do tensor, mas podemos especificar o eixo 
                        desejado como um parâmetro.''')
            st.code('''
                        import torch
                        # Cria 1 tensor de 2 dimensões
                        x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                    
                        # Soma total
                        r = torch.sum(x)         
                        print(r)
                    
                        # Soma por coluna
                        r = torch.sum(x, 0)         
                        print(r)
                    
                        # Soma por linha
                        r = torch.sum(x, 1)         
                        print(r)
                    ''',language='python')
            st.write('**Soma total**')
            st.write(torch.sum(x))
            st.write('**Soma por coluna**')
            st.write(torch.sum(x,0))
            st.write('**Soma por linha**')
            st.write(torch.sum(x,1))

        st.write('## **Multiplicação de Matrizes**')
        st.write('''
                    Vamos explorar três tipos fundamentais de multiplicação de 
                    matrizes em PyTorch: multiplicação elemento a elemento (element-wise), 
                    produto escalar (dot product) e produto vetorial (cross product).
               ''')
        st.write('### Multiplicação Element-wise em PyTorch:')
        st.write('''
                 A multiplicação elemento a elemento em PyTorch é realizada diretamente 
                 usando o operador de multiplicação (*) ou pelo método `torch.mul()`. Esta operação é aplicada entre 
                 dois tensores de mesma forma e resulta em um novo tensor com os elementos 
                 multiplicados elemento a elemento. A multiplicação elemento a elemento é útil 
                 em muitas aplicações, incluindo operações de ativação em redes neurais e operações 
                 ponto a ponto em processamento de sinais.''')
        st.image('imagens/elementwise.jpg')
        
        with st.expander('**Multiplicação Element-wise**'):
             x = torch.Tensor([[1,2],[3,4]])
             y = torch.Tensor([[5,2],[4,5]])
             r = torch.mul(x, y)
             st.code('''
                        # Criando os tensores
                        x = torch.Tensor([[1,2],[3,4]])
                        y = torch.Tensor([[5,2],[4,5]])
                     
                        # Multiplicando tensor x * y    
                        r = torch.mul(x, y) 
                        print(r)
                    ''',language='python')
             st.write('**Resultado da multiplicação tensor x*y (Element-Wise)**')
             st.write(r)
        st.write('### Multiplicação Dot Product')
        st.write('''
                    O produto escalar ou multiplicação Dot Product em PyTorch é realizado usando a função `torch.dot()`. 
                    Esta função calcula o produto escalar entre dois tensores unidimensionais 
                    (vetores), multiplicando seus elementos correspondentes e somando-os. 
                    O produto escalar é comumente usado em cálculos de similaridade, projeções 
                    e otimização de modelos de aprendizado de máquina.
                ''')
        st.image('imagens/dotproduct.png')
        with st.expander('**Multiplicação Dot Product**'):
             t1 = torch.Tensor([4,2])
             t2 = torch.Tensor([3,1])
             r = torch.dot(t1, t2)
             st.code('''
                        # Criando dois tensores
                        t1 = torch.Tensor([4,2])
                        t2 = torch.Tensor([3,1])

                        # Multiplicando os tensores.
                        r = torch.dot(t1, t2)
                        print(r)   

                        ''', language='python')
             st.write('**Produto escalar t1 * t2 (Dot Product)**')
             st.write(r)

        st.write('### Multiplicação Cross Product')
        st.write('''A multiplicação Cross product é um produto de uma  multiplicação cruzada
                    onde podemos multiplicar matrizes e vetores com dimensões diferentes.''')

        with st.expander('**Multiplicação de Matriz por Vetor `torch.mv`**'):
             st.write('Nesta operação, multiplicamos uma matriz por um vetor para obter um novo vetor.')
             mat = torch.randn(2, 4)
             vec = torch.randn(4)
             st.code('''
                        # Criando Matriz e Tensor
                        mat = torch.randn(2, 4)
                        vec = torch.randn(4)
                        print(mat)
                        print(vec)
                     
                        #Multiplicando a matriz e vetor
                        r = torch.mv(mat, vec)
                        print(r)
                    ''',language='python')

             st.write('**Matriz**')
             st.write(mat)
             st.write('**Vetor**')
             st.write(vec)

             st.write('**Multiplicação entre Matriz e Vetor**')
             r = torch.mv(mat, vec)
             st.write(r)
             st.write('Também é possivel efetuar uma soma ao multiplicarmos uma Matriz x Vetor `torch.addmv()`')
             st.code('''
                        # Multiplicação entre Matriz e Vetor e ao resultado somamos outro vetor.
                        V = torch.randn(2)
                        mat = torch.randn(2, 3)
                        vec = torch.randn(3)
                        
                        # Vetor + (Matriz X Vetor)
                        r = torch.addmv(V, mat, vec)
                        print(r)
                        
                    ''',language='python')
             V = torch.randn(2)
             mat = torch.randn(2, 3)
             vec = torch.randn(3)
             r = torch.addmv(V, mat, vec)
             st.write('**Vetor + (Matriz X Vetor)**')
             st.write(r)

        with st.expander('**Multiplicação entre matrizes - Cross Product `torch.cross`**'):
             st.write('''Essa função aceita dois tensores com a mesma 
                      forma e calcula o produto cruzado entre eles, produzindo um novo tensor''')
             st.code('''
                        # # Multiplicação entre Matrizes com produto cruzado (cross product)
                        # Matriz X Matriz
                        
                        m1 = torch.rand(3, 5)
                        m2 = torch.rand(3, 5)
                        r = torch.cross(m1, m2)
                     
                        # Resultado Size 3x5
                        print(r)
                        print(r.size())

                    ''',language='python')
             m1 = torch.rand(3, 5)
             m2 = torch.rand(3, 5)
             r = torch.cross(m1, m2)
             st.write('**Tensor 1**')   
             st.write(m1)
             st.write('**Tensor 2**') 
             st.write(m2)
             st.write('**Resultado Multiplicação Cross Product**')
             st.write(r)

    if selected == '4 - Concatenação, Expansão, Junção, Chunk, Squeeze':
         st.write('# Manipulação de Tensores.')
         st.write('Existem diversas formas de manipular e tensores:')
         with st.expander('**Expansão`torch.expand()`**'):
              st.write('''A expansão, também conhecida como broadcasting, é uma operação fundamental
                        em PyTorch que permite realizar operações entre tensores de diferentes formas, 
                       ajustando automaticamente as dimensões dos tensores menores para que sejam 
                       compatíveis com as dimensões dos tensores maiores. Essa operação é especialmente 
                       útil quando precisamos realizar operações entre tensores de formas diferentes sem 
                       precisar criar cópias adicionais dos dados.''')
              
              x = torch.tensor([[1],[2],[3]])
              st.code(''' 
                        # Criando um tensor - size 3x1
                        x = torch.tensor([[1],[2],[3]])
                        print(x)
                      ''',language='python')
              st.write('**Tensor - Size 3x1**')
              st.write(x.numpy())

              st.code('''
                      # Expandindo um tensor
                      x.expand(3, 4)
                      ''',language='python')
              st.write(x.expand(3, 4).numpy())

         with st.expander('**Concatenação `torch.cat()`**'):
              st.write(''' A concatenação é uma operação que permite combinar tensores
                        ao longo de um eixo específico. Isso é útil para combinar 
                       dados de diferentes fontes ou para aumentar o tamanho de um tensor''')
              st.code('''
                        # Criando tensor - Size 5x3
                        x = torch.randn(5, 3).type(torch.FloatTensor)
                        print(x)
                        ''',language='python')
              x = torch.randn(5, 3).type(torch.FloatTensor)
              st.write('**Tensor x - Size 5x3**')
              st.write(x.numpy())
              
              st.code('''
                        # Concatenação por linha
                        x_row = torch.cat((x, x, x), 0)
                        print(x_row)  
                      ''',language='python')
              st.write('**Concatenação por linha**')  
              st.write(torch.cat((x, x, x), 0).numpy())
              
              st.code('''
                        # Concatenação por coluna
                        x_col = torch.cat((x, x, x), 1)
                        print(x_col)  
                      ''',language='python')
              st.write('**Concatenação por coluna**')  
              st.write(torch.cat((x, x, x), 1).numpy())

         with st.expander('**Junção (Stacking) `torch.stack()`**'):
              st.write('''A junção é uma operação que permite empilhar tensores ao longo 
                       de um novo eixo. Isso é útil para combinar múltiplos tensores em 
                       um único tensor multidimensional. ''')
                                  
              
             

            
        
if __name__ == "__main__":
    main()
