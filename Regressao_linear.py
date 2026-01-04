"""
ANÁLISE DE REGRESSÃO LINEAR PARA TERMOPARES

Created on Wed Oct 06 14:32:12 2025
@author: Nicole

Objetivo: Analisar e comparar dados de diferentes termopares
"""

# Bibliotecas
import numpy as np  # Para cálculos matemáticos com arrays
import matplotlib.pyplot as plt  # Para criar gráficos

class Termopar:
    # Classe que representa um tipo de termopar.
    # Cada termopar tem dados de temperatura e tensão.
    # Aqui calculamos a regressão linear e métricas.

    def __init__(self, nome, temperatura, tensao):
        # Inicializa um termopar com seus dados
        
        # Nome: tipo do termopar
        # Temperatura: array com valores de temperatura em °C
        # Tensão: array com valores de tensão correspondentes em mV

        self.nome = nome  # Nome do termopar
        self.T = np.array(temperatura)  # Temperaturas como array numpy
        self.V = np.array(tensao)  # Tensões como array numpy
        self.n = len(self.T)  # Número de medições
        
        # Variáveis que serão calculadas depois
        self.a = 0  # Coeficiente angular da reta (a)
        self.b = 0  # Coeficiente linear da reta (b)
        self.r2 = 0  # Coeficiente de determinação (R²)
        self.rmse = 0  # Raiz do erro quadrático médio
        self.V_previsto = None  # Valores previstos pela regressão
        self.residuos = None  # Diferenças entre valores reais e previstos
    
    def calcular_regressao(self):
        """
        Calcula a regressão linear usando método dos mínimos quadrados
        Encontra a melhor reta que aproxima os dados
        """
        # Passo 1: Calcular somatórios necessários
        T_soma = np.sum(self.T)
        T2_soma = np.sum(self.T**2)
        V_soma = np.sum(self.V)
        TV_soma = np.sum(self.T * self.V)
        
        # Coeficiente angular (a) - inclinação da reta
        numerador_a = self.n * TV_soma - T_soma * V_soma
        denominador_a = self.n * T2_soma - T_soma**2
        self.a = numerador_a / denominador_a
        self.b = (V_soma - self.a * T_soma) / self.n # Coeficiente linear (b) - onde a reta corta o eixo Y
        self.V_previsto = self.a * self.T + self.b # Para cada temperatura T, calculamos V_previsto = a*T + b
        self.residuos = self.V - self.V_previsto # Resíduo (erro) = Valor real - Valor previsto
    
    def calcular_metricas(self):
        # Calcula métricas para avaliar a qualidade da regressão
        # R²: quanto da variação é explicada pela reta (0 a 1, quanto maior melhor)
        # RMSE: erro médio em mV (quanto menor melhor)

        if self.V_previsto is None:
            self.calcular_regressao()
        
        # Cálculo do R² (Coeficiente de Determinação)
        # 1. Média das tensões reais
        V_media = np.mean(self.V)
        
        # Soma total dos quadrados (variação total dos dados)
        SS_tot = np.sum((self.V - V_media)**2)
        
        # Soma dos quadrados dos resíduos (variação não explicada)
        SS_res = np.sum((self.V - self.V_previsto)**2)
        
        # 4. R² = 1 - (variação não explicada / variação total)
        self.r2 = 1 - (SS_res / SS_tot)
        
        # Cálculo do RMSE (Raiz do Erro Quadrático Médio)
        mse = np.mean((self.residuos)**2) # Erro quadrático médio (média dos quadrados dos resíduos)
        self.rmse = np.sqrt(mse) # Raiz quadrada do MSE
    
    def plot_dados_e_regressao(self):
        # Gráfico que mostra os dados e a linha de regressão
        plt.figure(figsize=(10, 6))
        
        # Plotando os dados reais (pontos)
        plt.scatter(self.T, self.V, color='blue', s=50, 
                   label=f'Dados {self.nome} (n={self.n})', 
                   alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plotando a linha de regressão
        T_suave = np.linspace(min(self.T), max(self.T), 100)
        V_suave = self.a * T_suave + self.b
        
        plt.plot(T_suave, V_suave, 'r-', linewidth=3, 
                label=f'Regressão: V = {self.a:.4f}T + {self.b:.4f}')
        
        # Configurando o gráfico
        plt.xlabel('Temperatura [°C]', fontsize=12)
        plt.ylabel('Tensão [mV]', fontsize=12)
        plt.title(f'Termopar {self.nome} - Regressão Linear\nR² = {self.r2:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuos_vs_temperatura(self):
        # Cria gráfico de resíduos (erros) vs temperatura
        # Mostra se os erros são aleatórios ou seguem algum padrão
        if self.residuos is None:
            self.calcular_regressao()
            self.calcular_metricas()
        
        plt.figure(figsize=(10, 5))
        
        # Plotar resíduos
        plt.scatter(self.T, self.residuos, color='purple', s=50, 
                   alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Linha horizontal em y = 0 (referência)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Configurar o gráfico
        plt.xlabel('Temperatura [°C]', fontsize=12)
        plt.ylabel('Resíduos [mV]', fontsize=12)
        plt.title(f'Termopar {self.nome} - Análise de Resíduos\nRMSE = {self.rmse:.4f} mV', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_distribuicao_residuos(self):
        #Cria histograma mostrando a distribuição dos resíduos
        #Ideal: distribuição normal em torno de zero
        
        if self.residuos is None:
            self.calcular_regressao()
            self.calcular_metricas()
        
        plt.figure(figsize=(10, 5))
        
        # Histograma dos resíduos
        plt.hist(self.residuos, bins=12, color='green', alpha=0.7, 
                edgecolor='black', linewidth=1)
        
        # Linha vertical em x = 0
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Configurar o gráfico
        plt.xlabel('Resíduos [mV]', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title(f'Termopar {self.nome} - Distribuição dos Resíduos\nMédia = {np.mean(self.residuos):.4f} mV', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def resumo(self):
        # Retorna um resumo das características do termopar
        # Usado para criar a tabela comparativa
        
        return {
            'Termopar': self.nome,
            'Equação': f"V = {self.a:.4f}T + {self.b:.4f}",
            'R²': self.r2,
            'RMSE [mV]': self.rmse,
            'n': self.n
        }

def carregar_arquivo_txt(nome_arquivo):
# Carrega dados de um arquivo .txt e retorna: temperatura, tensao
    try: # Verificar se arquivo existe
        
        # Carregar dados do arquivo
        dados = np.loadtxt(nome_arquivo, delimiter=',')
        
        # Separar em temperatura (1ª coluna) e tensão (2ª coluna)
        temperatura = dados[:, 0]
        tensao = dados[:, 1]
        
        return temperatura, tensao
        
    except Exception as e:
        print(f"  ERRO ao carregar {nome_arquivo}: {str(e)}")
        return None, None

def criar_termopar_de_arquivo(nome_arquivo):
    # Cria um objeto Termopar a partir de um arquivo .txt
    # Extrai o nome do termopar do nome do arquivo
    
    # Extrair nome do termopar do nome do arquivo
    nome = nome_arquivo.replace('Tabela_', '').replace('.txt', '').upper()
    
    # Carregar dados do arquivo
    T, V = carregar_arquivo_txt(nome_arquivo)
    
    if T is not None:
        # Criar objeto Termopar
        termopar = Termopar(nome, T, V)
        
        # Calcular regressão e métricas
        termopar.calcular_regressao()
        termopar.calcular_metricas()
        
        return termopar
    else:
        return None

#Funções para visualização e análise
def mostrar_tabela_comparativa(termopares): #Tabela comparativa de todos os termopares analisados
    print("\n" + "="*73)
    print("                   TABELA COMPARATIVA DE TERMOPARES")
    print("="*73)
    
    # Cabeçalho da tabela
    print(f"{'Termopar':<10} {'Equação':<35} {'R²':<10} {'RMSE [mV]':<12} {'Amostras':<6}")
    print("-" * 73)
    
    # Linhas da tabela (um por termopar)
    for tp in termopares:
        resumo = tp.resumo()
        print(f"{resumo['Termopar']:<10} {resumo['Equação']:<35} "
              f"{resumo['R²']:<10.4f} {resumo['RMSE [mV]']:<12.4f} {resumo['n']:<6}")
    
    print("="*73)
    
    # Encontrar o melhor termopar (maior R²)
    melhor_r2 = max([tp.r2 for tp in termopares])
    for tp in termopares:
        if tp.r2 == melhor_r2:
            print(f"\nMELHOR AJUSTE LINEAR: Termopar {tp.nome}")
            print(f"   R² = {tp.r2:.4f} ({tp.r2*100:.1f}% da variação explicada)")
            print(f"   RMSE = {tp.rmse:.4f} mV (precisão média)")
            break

def plot_comparativo_regressoes(termopares): # Plota todas as regressões em um único gráfico para comparação visual
    plt.figure(figsize=(12, 7))
    
    # Cores diferentes para cada termopar
    cores = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plotar cada termopar
    for i, tp in enumerate(termopares):
        cor = cores[i % len(cores)]  # Escolher cor (se muitos, repete)
        
        # Plota dados (pontos)
        plt.scatter(tp.T, tp.V, color=cor, s=40, 
                   label=f'{tp.nome} (R²={tp.r2:.3f})',
                   alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Plota linha de regressão
        T_suave = np.linspace(min(tp.T), max(tp.T), 100)
        V_suave = tp.a * T_suave + tp.b
        plt.plot(T_suave, V_suave, color=cor, linewidth=2, alpha=0.8)
    
    # Configurar o gráfico
    plt.xlabel('Temperatura [°C]', fontsize=12)
    plt.ylabel('Tensão [mV]', fontsize=12)
    plt.title('Comparativo: Regressões Lineares de Diferentes Termopares', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.show()

def analisar_todos_termopares(lista_arquivos): # Função principal: analisa todos os termopares da lista
    print("="*34)
    print("ANÁLISE COMPARATIVA DE TERMOPARES")
    print("="*34)
    
    termopares = []  # Lista para armazenar todos os termopares
    
    # Carregar e analisar cada termopar
    for arquivo in lista_arquivos:
        tp = criar_termopar_de_arquivo(arquivo)
        
        if tp is not None:
            termopares.append(tp)
            
            # Mostrar resultados individuais
            print(f" Termopar tipo {tp.nome}:")
            print(f"    Equação: V = {tp.a:.4f}T + {tp.b:.4f}")
            print(f"    R² = {tp.r2:.4f}")
            print(f"    RMSE = {tp.rmse:.4f} mV")
            print(f"    n = {tp.n} pontos")
            print("-"*34)
    
    # Gerar gráficos individuais para cada termopar
    for tp in termopares:
        # Gráfico 1: Dados e regressão
        tp.plot_dados_e_regressao()
        
        # Gráfico 2: Resíduos vs Temperatura
        tp.plot_residuos_vs_temperatura()
        
        # Gráfico 3: Distribuição dos resíduos
        tp.plot_distribuicao_residuos()

    mostrar_tabela_comparativa(termopares)
    plot_comparativo_regressoes(termopares)

def main(): # Função principal que executa todo o programa
    arquivos_termopares = [
        'Tabela_N.txt',
        'Tabela_J.txt', 
        'Tabela_K.txt',
        'Tabela_B.txt',
        'Tabela_R.txt',
        'Tabela_E.txt'
    ]
    
    # Executa a análise completa
    analisar_todos_termopares(arquivos_termopares)

# Inicia o programa
main()