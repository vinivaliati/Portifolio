import matplotlib.pyplot as plt
import seaborn as sns
import math


def compare_variables(df, target):
    """
    Gera boxplots para variáveis categóricas (object) e scatterplots para variáveis numéricas
    (int64, float64) comparando-as com a variável alvo (target).
    Os gráficos são organizados com até 5 gráficos por linha.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        target (str): Nome da variável de resposta.
    """
    # Separar variáveis categóricas (object) e numéricas (int64, float64)
    categorical_vars = df.select_dtypes(include=['object']).columns
    numerical_vars = df.select_dtypes(include=['int64', 'float64']).columns

    # Verificar se a variável alvo está no DataFrame
    if target not in df.columns:
        raise ValueError(f"The target variable '{target}' is not in the DataFrame.")

    # Criar gráficos categóricos
    num_categorical_vars = len([var for var in categorical_vars if var != target])
    num_rows_cat = math.ceil(num_categorical_vars / 5)
    fig_cat, axes_cat = plt.subplots(num_rows_cat, 5, figsize=(20, 5 * num_rows_cat))
    axes_cat = axes_cat.flatten()
    
    for i, var in enumerate(categorical_vars):
        if var != target:
            sns.boxplot(x=df[var], y=df[target], ax=axes_cat[i])
            axes_cat[i].set_title(f'{var} vs {target}')
            axes_cat[i].tick_params(axis='x', rotation=45)
            axes_cat[i].set_xlabel(var)
            axes_cat[i].set_ylabel(target)
    # Remover eixos extras
    for ax in axes_cat[num_categorical_vars:]:
        fig_cat.delaxes(ax)

    plt.tight_layout()
    plt.show()

    # Criar gráficos numéricos
    num_numerical_vars = len([var for var in numerical_vars if var != target])
    num_rows_num = math.ceil(num_numerical_vars / 5)
    fig_num, axes_num = plt.subplots(num_rows_num, 5, figsize=(20, 5 * num_rows_num))
    axes_num = axes_num.flatten()
    
    for i, var in enumerate(numerical_vars):
        if var != target:
            sns.scatterplot(x=df[var], y=df[target], ax=axes_num[i])
            axes_num[i].set_title(f'{var} vs {target}')
            axes_num[i].set_xlabel(var)
            axes_num[i].set_ylabel(target)
    # Remover eixos extras
    for ax in axes_num[num_numerical_vars:]:
        fig_num.delaxes(ax)

    plt.tight_layout()
    plt.show()



def plot_correlation_with_target(df, target):
    """
    Gera um gráfico de barras mostrando a correlação das colunas numéricas de um DataFrame 
    com uma coluna-alvo específica.
    
    Parâmetros:
        df (DataFrame): O DataFrame contendo os dados.
        target_column (str): O nome da coluna-alvo para calcular as correlações.
    """
    # Seleciona apenas colunas numéricas (int e float)
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calcula a correlação com a coluna-alvo
    if target not in numeric_df:
        raise ValueError(f"A coluna '{target}' não é numérica ou não está no DataFrame.")
    
    correlation = numeric_df.corr()[target].sort_values()
    
    # Configura o tamanho do gráfico
    plt.figure(figsize=(10, 8))
    
    # Plota o gráfico de barras
    ax = sns.barplot(x=correlation.index, y=correlation.values, color='skyblue')
    
    # Adiciona os valores nas colunas
    for i, value in enumerate(correlation.values):
        plt.text(i, value + 0.01 if value > 0 else value - 0.01,  # Ajusta posição do texto
                 f'{value:.2f}', ha='center', va='bottom' if value > 0 else 'top', fontsize=10)
    
    # Remove gridlines
    ax.grid(False)
    
    # Configurações estéticas
    plt.xticks(rotation=90)
    plt.title(f"Correlação com {target}")
    plt.ylabel("Coeficiente de Correlação")
    plt.xlabel("Variáveis")
    plt.tight_layout()
    plt.show()



def plot_target_correlation_heatmap(df, target):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if target not in numeric_df:
        raise ValueError(f"A coluna '{target}' não é numérica ou não está no DataFrame.")
    correlation = numeric_df.corr()[[target]].sort_values(by=target)
    plt.figure(figsize=(6, len(correlation) * 0.5))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=False
    )
    plt.title(f"Heatmap de Correlação com '{target}'")
    plt.xlabel("Correlação")
    plt.ylabel("Variáveis")
    plt.tight_layout()
    plt.show()