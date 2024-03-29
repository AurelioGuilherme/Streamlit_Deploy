import streamlit as st
import torch
import pytorch_lightning as pl
from streamlit_option_menu import option_menu
import numpy as np


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
                               '3 - Operações aritméticas com Tensores'], 
                           menu_icon="", default_index=0)

st.write("[Conheça o meu GitHub](https://github.com/AurelioGuilherme)")
st.write("[Documentação PyTorch](https://pytorch.org/docs/stable/index.html)")


def main():
    if selected == '0 - O que são tensores?':
        st.write('# **O que são tensores?**')
        st.write("""
        O objeto tensor utilizado no framework PyTorch é focado e projetado para processamento paralelizado. 
        
        Ele representa uma estrutura de dados multidimensional que pode ser manipulada de forma eficiente em GPUs e CPUs. 
        
        Essa capacidade de processamento paralelo é essencial para lidar com grandes conjuntos de dados e realizar cálculos complexos de forma eficiente. 
        
        Os tensores no PyTorch são a base para a construção de modelos de aprendizado profundo e outras tarefas de computação científica, 
        fornecendo uma maneira flexível e eficaz de representar e manipular dados em várias dimensões.
        """)

        st.image('imagens/tensor.png')

        st.write("""
        Os tensores são amplamente utilizados em várias áreas, incluindo aprendizado de máquina, visão computacional, processamento de linguagem natural e física, entre outros. 
        
        Eles fornecem uma maneira flexível e eficiente de representar e manipular dados em várias dimensões. 
        
        No aprendizado de máquina, os tensores são usados para representar conjuntos de dados, parâmetros de modelo, gradientes durante o treinamento e resultados de predição. 
        
        Eles são a base para a construção de modelos de aprendizado profundo, como redes neurais convolucionais e redes neurais recorrentes. 
        
        Além disso, em computação científica, os tensores são usados para representar tensores de tensão em mecânica, campos vetoriais em física e muito mais. 
        Sua capacidade de processamento paralelo em hardware especializado os torna essenciais para lidar com grandes volumes de dados e realizar cálculos complexos de forma eficiente.
        """)
        st.image('imagens/img-1.png')

        st.write("""
        Os tensores têm uma ampla gama de aplicações práticas em diversas áreas. 
                 
        No processamento de imagens, os tensores são usados para representar imagens digitais em três ou mais dimensões (largura, altura e canais de cor). 
        
        Isso permite realizar operações como convoluções e pooling em imagens para tarefas como classificação e detecção de objetos. 
        
        Em processamento de linguagem natural, os tensores são usados para representar sequências de palavras em texto e realizar operações como embeddings e atenção em modelos de processamento de linguagem. 
        
        Além disso, em física e engenharia, os tensores são usados para representar grandezas físicas como tensão, deformação e fluxo de calor em sistemas complexos.
        """)
        st.image('imagens/April-28-deep-learning-applications-infograph.png')
        
    elif selected == '1 - Criando Tensores':
        st.write('# **Criando Tensores**')
        st.write("### Tensores PyTorch x Arrays NumPy:\n")

        st.write("""
        Os tensores no PyTorch são estruturas de dados multidimensionais que podem ter uma ou mais dimensões. As dimensões de um tensor representam a forma ou o tamanho de cada eixo do tensor. Por exemplo, um tensor bidimensional tem duas dimensões: uma dimensão para linhas e outra para colunas. 
        """)

        st.write('Você pode criar tensores a partir de listas ou matrizes numpy e vice-versa utilizando a função `.Tensor()`')
        st.write('Esta função cria um objeto do tipo tensor')
        lista_python = [[1,2,3], [4,5,6]]
        t1 = torch.Tensor(lista_python)  

        # Mostrar o código
        with st.expander("Cria um tensor 2x3 a partir de uma lista Python"):
            st.code("""
                        import torch

                        # Cria um tensor 2x3 a partir de uma lista Python
                        lista_python = [[1,2,3], [4,5,6]]
                        t1 = torch.Tensor(lista_python)
                        print(t1)
                    """, language="python")
            st.write(t1)

        array_numpy = np.array([[9,6,1], [5,3,2]])
        t2 = torch.Tensor(array_numpy)

        # Mostrar código
        with st.expander('Cria um tensor 2x3 a partir de array numpy'):
            st.code('''
                        import torch
                        import numpy as np

                        # Cria um tensor 2x3 a partir de um array Numpy                    
                        array_numpy = np.array([[9,6,1], [5,3,2]])
                        t2 = torch.Tensor(array_numpy)
                        print(t2)                    
                    ''', language='python')
            st.write(t2)

        st.write('### Tipos de Tensores no PyTorch')
        st.write('''
        Ao criar tensores, podemos especificar suas dimensões e inicializá-los com diferentes valores. Por exemplo, podemos criar um tensor de zeros, onde todos os elementos do tensor têm o valor zero. Isso é útil para inicializar tensores antes de realizar operações ou preenchê-los com dados reais posteriormente. Podemos criar um tensor de zeros usando a função `torch.zeros(dimensões)`, onde "dimensões" é uma lista ou tupla que especifica o tamanho de cada dimensão do tensor.

        No entanto, é importante ter cuidado ao trabalhar com tensores e gerenciar a memória corretamente. Às vezes, ao criar ou manipular tensores, podemos gerar `"lixo de memória"`, que são áreas de memória alocadas para objetos que não estão mais em uso, mas ainda não foram liberadas. Isso pode levar a vazamentos de memória e redução do desempenho do programa. Para evitar o lixo de memória, é importante liberar os recursos adequadamente após o uso, usando métodos como "del" em tensores ou utilizando o mecanismo de coleta de lixo do Python.
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
        with st.expander("Criando um tensor de 1's semelhante (like) ao tensor de zeros (mesmas dimensões)"):
            st.code('''
                        import torch

                        # Criando um tensor de 1's semelhante (like) ao tensor de zeros (mesmas dimensões)             
                        d = torch.ones_like(a)
                        print(d)                    
                    ''', language='python')
            st.write(d, unsafe_allow_html=True)

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
                        

          

if __name__ == "__main__":
    main()
