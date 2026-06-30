# Importa a biblioteca copy, usada para duplicar objetos.
# Isso é útil para criar uma cópia do modelo base para cada cliente.
import copy

# Importa warnings para silenciar avisos do scikit-learn durante o treino.
import warnings

# Importa o NumPy, usado para trabalhar com vetores, matrizes e médias.
import numpy as np

# Importa o FederatedDataset do Flower Datasets.
# Ele será usado para baixar e particionar o Iris.
from flwr_datasets import FederatedDataset

# Importa o particionador IID.
# IID significa que os dados serão divididos de forma aproximadamente independente
# e identicamente distribuída entre os clientes.
from flwr_datasets.partitioner import IidPartitioner

# Importa o modelo de regressão logística do scikit-learn.
# Esse será o modelo treinado de forma federada.
from sklearn.linear_model import LogisticRegression

# Importa a função de log loss para medir a perda do modelo.
from sklearn.metrics import log_loss


# Lista com os nomes das colunas de entrada do Iris que serão usadas como atributos.
FEATURES = [
    "petal_length",
    "petal_width",
    "sepal_length",
    "sepal_width",
]


# Define uma função para carregar o dataset Iris e dividi-lo entre vários clientes.
def carregar_clientes(num_clientes):
    # Cria um particionador IID com a quantidade de clientes desejada.
    partitioner = IidPartitioner(num_partitions=num_clientes)

    # Cria o objeto FederatedDataset.
    # O dataset usado é o hitorilabs/iris, como no tutorial do Flower.
    # A chave "train" diz que o split "train" será particionado.
    fds = FederatedDataset(
        dataset="hitorilabs/iris",
        partitioners={"train": partitioner},
    )

    # Carrega o split completo "train".
    # O método with_format("pandas") faz com que o retorno seja tratado como DataFrame.
    # O [:] pega todo o conteúdo.
    df_completo = fds.load_split("train").with_format("pandas")[:]

    # Seleciona uma linha de cada classe.
    # Isso será usado apenas para inicializar corretamente o modelo do scikit-learn.
    # O groupby separa por espécie e o head(1) pega a primeira linha de cada classe.
    seed_rows = df_completo.groupby("species", sort=False).head(1)

    # Extrai as features dessas linhas-semente e converte para NumPy.
    X_seed = seed_rows[FEATURES].to_numpy(dtype=np.float64)

    # Extrai os rótulos dessas linhas-semente.
    y_seed = seed_rows["species"].to_numpy()

    # Cria uma lista vazia que armazenará os dados locais de cada cliente.
    clientes = []

    # Percorre todos os clientes simulados.
    for cid in range(num_clientes):
        # Carrega a partição do cliente atual.
        df_particao = fds.load_partition(cid, "train").with_format("pandas")[:]

        # Extrai as features da partição.
        X = df_particao[FEATURES].to_numpy(dtype=np.float64)

        # Extrai os rótulos da partição.
        y = df_particao["species"].to_numpy()

        # Calcula o ponto de corte para separar 80% treino e 20% teste.
        split_idx = int(0.8 * len(X))

        # Garante que exista pelo menos 1 amostra de treino.
        if split_idx <= 0:
            split_idx = 1

        # Garante que exista pelo menos 1 amostra de teste.
        if split_idx >= len(X):
            split_idx = len(X) - 1

        # Separa as amostras de treino do cliente.
        X_train = X[:split_idx]

        # Separa os rótulos de treino do cliente.
        y_train = y[:split_idx]

        # Separa as amostras de teste do cliente.
        X_test = X[split_idx:]

        # Separa os rótulos de teste do cliente.
        y_test = y[split_idx:]

        # Guarda tudo em um dicionário para facilitar o acesso.
        cliente = {
            "cid": cid,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        # Adiciona o cliente na lista.
        clientes.append(cliente)

    # Retorna:
    # 1) a lista de clientes,
    # 2) as features-semente,
    # 3) os rótulos-semente.
    return clientes, X_seed, y_seed


# Define uma função para criar o modelo inicial.
def criar_modelo_inicial(X_seed, y_seed):
    # Cria a regressão logística.
    # max_iter=1 faz cada treino local ser curto.
    # warm_start=True faz o modelo continuar a partir dos pesos atuais.
    # solver="saga" é o mesmo tipo de escolha comum no tutorial do Flower.
    # random_state deixa os resultados mais reprodutíveis.
    modelo = LogisticRegression(
        penalty="l2",
        max_iter=1,
        warm_start=True,
        solver="saga",
        random_state=42,
    )

    # O scikit-learn precisa de um fit inicial para criar atributos internos
    # como classes_, coef_ e intercept_.
    # Vamos fazer esse fit só para "inicializar" o objeto corretamente.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(X_seed, y_seed)

    # Depois da inicialização, zeramos os pesos.
    # Isso faz a federação começar de um estado neutro.
    modelo.coef_ = np.zeros_like(modelo.coef_, dtype=np.float64)

    # Também zeramos o termo de bias/intercepto.
    modelo.intercept_ = np.zeros_like(modelo.intercept_, dtype=np.float64)

    # Retorna o modelo inicializado.
    return modelo


# Define uma função para pegar os parâmetros do modelo.
def obter_parametros(modelo):
    # Retorna uma lista com:
    # 1) a matriz de pesos
    # 2) o vetor de interceptos
    return [modelo.coef_.copy(), modelo.intercept_.copy()]


# Define uma função para colocar parâmetros em um modelo.
def definir_parametros(modelo, parametros):
    # Copia a matriz de pesos para dentro do modelo.
    modelo.coef_ = parametros[0].copy()

    # Copia o vetor de interceptos para dentro do modelo.
    modelo.intercept_ = parametros[1].copy()


# Define a função que simula o treino local de um cliente.
def treino_local(cliente, parametros_globais, modelo_base, epocas_locais=1):
    # Cria uma cópia do modelo base.
    # Assim, cada cliente trabalha no seu próprio modelo local.
    modelo = copy.deepcopy(modelo_base)

    # Coloca no modelo local os parâmetros globais recebidos do servidor.
    definir_parametros(modelo, parametros_globais)

    # Conta quantas classes diferentes existem no treino local do cliente.
    num_classes_locais = len(np.unique(cliente["y_train"]))

    # Só treinamos se houver pelo menos 2 classes no treino local.
    # Isso evita erro no scikit-learn em casos muito extremos.
    if num_classes_locais >= 2:
        # Silencia avisos do scikit-learn durante o fit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Repete o treino local o número de épocas desejado.
            for _ in range(epocas_locais):
                modelo.fit(cliente["X_train"], cliente["y_train"])

    # Calcula probabilidades no conjunto de treino local.
    train_proba = modelo.predict_proba(cliente["X_train"])

    # Calcula a perda logarítmica no treino local.
    train_loss = log_loss(
        cliente["y_train"],
        train_proba,
        labels=modelo.classes_,
    )

    # Calcula a acurácia no treino local.
    train_acc = modelo.score(cliente["X_train"], cliente["y_train"])

    # Monta o resultado que será devolvido ao servidor.
    resultado = {
        "cid": cliente["cid"],
        "params": obter_parametros(modelo),
        "num_examples": len(cliente["X_train"]),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
    }

    # Retorna o resultado local do cliente.
    return resultado


# Define a função que agrega os parâmetros vindos de todos os clientes.
def agregar_fedavg(atualizacoes):
    # Soma o total de exemplos usados por todos os clientes.
    total_exemplos = sum(update["num_examples"] for update in atualizacoes)

    # Cria uma matriz de pesos zerada com o mesmo formato dos pesos do primeiro cliente.
    pesos_agregados = np.zeros_like(atualizacoes[0]["params"][0], dtype=np.float64)

    # Cria um vetor de interceptos zerado com o mesmo formato do primeiro cliente.
    interceptos_agregados = np.zeros_like(atualizacoes[0]["params"][1], dtype=np.float64)

    # Percorre todas as atualizações recebidas.
    for update in atualizacoes:
        # Calcula o peso relativo do cliente com base na quantidade de exemplos.
        peso_cliente = update["num_examples"] / total_exemplos

        # Soma a contribuição ponderada dos pesos do cliente.
        pesos_agregados += peso_cliente * update["params"][0]

        # Soma a contribuição ponderada dos interceptos do cliente.
        interceptos_agregados += peso_cliente * update["params"][1]

    # Retorna os parâmetros globais agregados.
    return [pesos_agregados, interceptos_agregados]


# Define uma função para avaliar o modelo global após cada rodada.
def avaliar_modelo_global(clientes, parametros_globais, modelo_base):
    # Cria uma cópia do modelo base.
    modelo = copy.deepcopy(modelo_base)

    # Coloca no modelo os parâmetros globais mais recentes.
    definir_parametros(modelo, parametros_globais)

    # Junta todos os conjuntos de teste dos clientes em um único conjunto global.
    X_test_global = np.concatenate([cliente["X_test"] for cliente in clientes], axis=0)

    # Junta todos os rótulos de teste dos clientes em um único vetor global.
    y_test_global = np.concatenate([cliente["y_test"] for cliente in clientes], axis=0)

    # Calcula as probabilidades previstas no teste global.
    test_proba = modelo.predict_proba(X_test_global)

    # Calcula a perda no teste global.
    test_loss = log_loss(
        y_test_global,
        test_proba,
        labels=modelo.classes_,
    )

    # Calcula a acurácia no teste global.
    test_acc = modelo.score(X_test_global, y_test_global)

    # Retorna as métricas calculadas.
    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "num_test_examples": len(X_test_global),
    }


# Define a função principal que executa toda a simulação federada.
def simular_aprendizado_federado(num_clientes=5, num_rodadas=5, epocas_locais=1):
    # Mostra uma mensagem indicando o início da preparação dos dados.
    print("Carregando e particionando o Iris com Flower Datasets...")

    # Carrega os clientes e os dados-semente.
    clientes, X_seed, y_seed = carregar_clientes(num_clientes)

    # Informa quantos clientes foram criados.
    print(f"{len(clientes)} clientes simulados foram criados.")

    # Percorre os clientes para mostrar o tamanho do conjunto local de cada um.
    for cliente in clientes:
        print(
            f"Cliente {cliente['cid']}: "
            f"{len(cliente['X_train'])} amostras de treino, "
            f"{len(cliente['X_test'])} amostras de teste"
        )

    # Cria o modelo global inicial.
    modelo_base = criar_modelo_inicial(X_seed, y_seed)

    # Extrai os parâmetros globais iniciais.
    parametros_globais = obter_parametros(modelo_base)

    # Linha em branco para melhorar a leitura da saída.
    print()

    # Mensagem de início das rodadas federadas.
    print("Iniciando rodadas federadas...")

    # Outra linha em branco para organizar melhor a saída.
    print()

    # Executa o número desejado de rodadas federadas.
    for rodada in range(1, num_rodadas + 1):
        # Cria uma lista vazia para guardar as atualizações locais da rodada.
        atualizacoes = []

        # Percorre todos os clientes simulados.
        for cliente in clientes:
            # Executa o treino local do cliente atual.
            resultado_local = treino_local(
                cliente=cliente,
                parametros_globais=parametros_globais,
                modelo_base=modelo_base,
                epocas_locais=epocas_locais,
            )

            # Guarda o resultado local na lista de atualizações.
            atualizacoes.append(resultado_local)

        # O servidor agrega as atualizações recebidas dos clientes.
        parametros_globais = agregar_fedavg(atualizacoes)

        # O servidor avalia o novo modelo global.
        metricas_globais = avaliar_modelo_global(
            clientes=clientes,
            parametros_globais=parametros_globais,
            modelo_base=modelo_base,
        )

        # Calcula a média da perda de treino entre os clientes.
        media_train_loss = sum(u["train_loss"] for u in atualizacoes) / len(atualizacoes)

        # Calcula a média da acurácia de treino entre os clientes.
        media_train_acc = sum(u["train_acc"] for u in atualizacoes) / len(atualizacoes)

        # Mostra o número da rodada atual.
        print(f"Rodada {rodada}/{num_rodadas}")

        # Mostra as métricas médias de treino local.
        print(
            f"  Treino médio local - "
            f"loss: {media_train_loss:.4f} | "
            f"acc: {media_train_acc:.4f}"
        )

        # Mostra as métricas do modelo global agregado.
        print(
            f"  Modelo global      - "
            f"loss: {metricas_globais['test_loss']:.4f} | "
            f"acc: {metricas_globais['test_acc']:.4f}"
        )

        # Mostra uma linha separadora para deixar a saída mais organizada.
        print("-" * 60)


# Este bloco garante que a simulação só será executada
# quando este arquivo for rodado diretamente.
# Se o arquivo for apenas importado em outro script, essa parte não roda.
if __name__ == "__main__":
    # Chama a função principal da simulação.
    # Aqui escolhemos:
    # - 5 clientes
    # - 5 rodadas federadas
    # - 1 época local por rodada
    simular_aprendizado_federado(
        num_clientes=5,
        num_rodadas=10,
        epocas_locais=1,
    )