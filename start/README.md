# Tranalho 1: cliente e servidor gRPC para aprendizado federado

Este material foi organizado para que os alunos implementem **passo a passo** um cliente e um servidor gRPC para uma simulação de aprendizado federado com o dataset Iris.

A ideia didática é simples:
- o **servidor** mantém o modelo global e controla as rodadas;
- cada **cliente** consulta o modelo global, treina localmente com sua partição do Iris e envia sua atualização;
- quando o servidor recebe todas as atualizações esperadas de uma rodada, ele faz a agregação e avança para a próxima.

O foco deste kit é guiá-los por pequenas etapas, com **TODOs claros** no código e **testes unitários progressivos**.

---

## 1. Estrutura do material

### Arquivos principais
- `federated.proto`
  - descreve as mensagens e os métodos RPC.
- `common_fl.py`
  - contém funções prontas de apoio para dataset, modelo, treino local, agregação e avaliação.
- `student_fl_client_grpc.py`
  - arquivo do cliente com `TODOs`.
- `student_fl_server_grpc.py`
  - arquivo do servidor com `TODOs`.

### Arquivos de apoio
- `tests/`
  - testes unitários que guiam a implementação.

---

## 2. Dependências

Instale as bibliotecas abaixo:

```bash
pip install grpcio grpcio-tools flwr-datasets scikit-learn numpy pandas
```

### Para que serve cada dependência?
- `grpcio`
  - fornece o runtime do gRPC em Python.
- `grpcio-tools`
  - gera os arquivos Python a partir do `.proto`.
- `flwr-datasets`
  - carrega e particiona o dataset Iris em vários clientes.
- `scikit-learn`
  - fornece o modelo `LogisticRegression` e métricas.
- `numpy`
  - manipula os parâmetros do modelo como arrays.
- `pandas`
  - é usado internamente ao converter partições do dataset para DataFrame.

---

## 3. Gerando os arquivos do protobuf

Antes de executar cliente ou servidor, gere os arquivos Python a partir do contrato gRPC:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. federated.proto
```

Isso deve criar:
- `federated_pb2.py`
- `federated_pb2_grpc.py`

> Sem esses arquivos, os imports `import federated_pb2` e `import federated_pb2_grpc` irão falhar.

---

## 4. O que já está pronto em `common_fl.py`

**Não precisam implementar** o arquivo `common_fl.py`. Ele existe para reduzir a complexidade e deixar o foco no protocolo federado via gRPC.

### Funções disponíveis

#### `load_client_partition(cid, num_clients)`
Retorna os dados locais do cliente:
```python
X_train, y_train, X_test, y_test = load_client_partition(cid, num_clients)
```

#### `create_model_template(num_clients)`
Cria o modelo-base local que será usado pelos clientes e pelo servidor:
```python
model_template = create_model_template(num_clients)
```

#### `local_train(X_train, y_train, global_params, model_template, local_epochs=1)`
Executa o treino local do cliente.
Retorna um dicionário com:
- `params`
- `num_examples`
- `train_loss`
- `train_acc`

Uso esperado:
```python
train_result = local_train(
    X_train=X_train,
    y_train=y_train,
    global_params=params,
    model_template=model_template,
    local_epochs=local_epochs,
)
```

#### `aggregate_fedavg(updates)`
Recebe uma lista de updates e retorna os parâmetros globais agregados:
```python
new_global_params = aggregate_fedavg(list(self.received_updates.values()))
```

#### `evaluate_global_model(global_params, model_template, X_test, y_test)`
Avalia o modelo global já agregado:
```python
loss, acc = evaluate_global_model(
    self.global_params,
    self.model_template,
    self.X_test_global,
    self.y_test_global,
)
```

#### `get_model_params(model)`
Extrai os parâmetros do modelo em formato `[coef, intercept]`.

#### `load_all_test_sets(num_clients)`
Junta os conjuntos de teste de todos os clientes para avaliação global.

---

## 5. Estratégia didática sugerida

A ordem recomendada é:

1. cliente: serialização dos parâmetros;
2. servidor: serialização e envio do modelo global;
3. servidor: validação das atualizações;
4. servidor: consolidação da rodada;
5. cliente: laço principal.

Isso foi pensado para que os testes comecem por funções pequenas e só depois avancem para o comportamento completo.

---

## 6. Implementação do cliente: o que fazer em cada função

Arquivo: `student_fl_client_grpc.py`

### 6.1 `params_from_proto(proto_params)`
Objetivo:
- converter a mensagem `ModelParameters` recebida do gRPC em arrays NumPy.

A estrutura esperada é:
- `coef`: matriz 2D
- `intercept`: vetor 1D

### Campos do protobuf que você deve usar
- `proto_params.coef_values`
- `proto_params.coef_shape`
- `proto_params.intercept_values`

### Passos sugeridos
1. transformar `coef_values` em `np.array(..., dtype=np.float64)`;
2. usar `.reshape(proto_params.coef_shape)` para reconstruir a matriz;
3. transformar `intercept_values` em `np.array(..., dtype=np.float64)`;
4. retornar `[coef, intercept]`.

---

### 6.2 `params_to_proto(params)`
Objetivo:
- fazer o caminho inverso: receber `[coef, intercept]` e criar `federated_pb2.ModelParameters(...)`.

### Campos que precisam ser preenchidos
- `coef_values`
- `coef_shape`
- `intercept_values`

### Passos sugeridos
1. separar `coef, intercept = params`;
2. usar `coef.ravel().tolist()` para achatar a matriz;
3. usar `list(coef.shape)` para guardar o formato;
4. usar `intercept.ravel().tolist()`;
5. retornar `federated_pb2.ModelParameters(...)`.

---

### 6.3 `build_client_update(cid, round_number, train_result)`
Objetivo:
- montar a mensagem `ClientUpdate` que será enviada ao servidor.

### Campos de `train_result` que você deve usar
- `train_result["params"]`
- `train_result["num_examples"]`
- `train_result["train_loss"]`
- `train_result["train_acc"]`

### Passos sugeridos
1. serializar `train_result["params"]` chamando `params_to_proto(...)`;
2. construir `federated_pb2.ClientUpdate(...)`;
3. preencher `cid`, `round`, `num_examples`, `train_loss`, `train_acc` e `model`.

---

### 6.4 `should_wait(global_round, completed_round)`
Essa função já está pronta.

Interpretação:
- se o servidor ainda está na mesma rodada que o cliente já terminou, o cliente não deve treinar de novo;
- ele deve apenas esperar e consultar novamente depois.

---

### 6.5 `run_client(...)`
Objetivo:
- implementar o laço principal do cliente.

### Ordem exata sugerida

#### Passo 1 — carregar os dados locais
Use:
```python
X_train, y_train, _, _ = load_client_partition(cid, num_clients)
```

#### Passo 2 — criar o modelo-base local
Use:
```python
model_template = create_model_template(num_clients)
```

#### Passo 3 — abrir o canal gRPC e criar o stub
Use:
```python
with grpc.insecure_channel(server_address) as channel:
    stub = federated_pb2_grpc.FederatedLearningStub(channel)
```

#### Passo 4 — pedir o modelo global ao servidor
Dentro do loop, use:
```python
global_model = stub.GetGlobalModel(federated_pb2.ClientHello(cid=cid))
```

#### Passo 5 — encerrar se o treinamento acabou
Se:
```python
global_model.done
```
for `True`, imprima uma mensagem e use `break`.

#### Passo 6 — esperar se não houver rodada nova
Use:
```python
if should_wait(global_model.round, completed_round):
    time.sleep(poll_interval)
    continue
```

#### Passo 7 — converter o modelo recebido
Use:
```python
global_params = params_from_proto(global_model.model)
```

#### Passo 8 — treinar localmente
Use:
```python
train_result = local_train(
    X_train=X_train,
    y_train=y_train,
    global_params=global_params,
    model_template=model_template,
    local_epochs=local_epochs,
)
```

#### Passo 9 — montar o update
Use:
```python
update = build_client_update(cid, global_model.round, train_result)
```

#### Passo 10 — enviar a atualização ao servidor
Use:
```python
ack = stub.SubmitUpdate(update)
```

#### Passo 11 — processar a resposta do servidor
Se:
```python
ack.accepted
```
for `True`:
- atualizar `completed_round = global_model.round`;
- imprimir `train_result["train_loss"]` e `train_result["train_acc"]`.

Caso contrário:
- imprimir `ack.message`.

#### Passo 12 — pequena espera antes da próxima consulta
Use:
```python
time.sleep(poll_interval)
```

### Erro comum a evitar
Não atualize `completed_round` antes de o servidor responder com `ack.accepted == True`.

---

## 7. Implementação do servidor: o que fazer em cada função

Arquivo: `student_fl_server_grpc.py`

### 7.1 `_params_to_proto(self, params)`
Objetivo:
- mesmo comportamento de `params_to_proto` do cliente.

### Dica
Você pode quase copiar a lógica do cliente.

---

### 7.2 `_proto_to_params(self, proto_params)`
Objetivo:
- mesmo comportamento de `params_from_proto` do cliente.

---

### 7.3 `GetGlobalModel(self, request, context)`
Objetivo:
- devolver ao cliente o modelo global atual, a rodada atual e o estado de término.

### Campos importantes do objeto `self`
- `self.current_round`
- `self.total_rounds`
- `self.global_params`

### Observação importante
Mesmo quando `done == True`, o teste espera que a resposta continue tendo `round` e `model` preenchidos.

---

### 7.4 `_validate_update(self, request)`
Objetivo:
- garantir que a atualização recebida seja aceitável antes de armazená-la.

### Regras que devem ser implementadas
1. Se o treinamento já terminou, rejeitar;
2. Se request.round for diferente de current_round, rejeitar;
3. Se o mesmo cliente já enviou atualização nesta rodada, rejeitar.

### Formato de retorno esperado
Uma tupla:
```python
(accepted, message)
```
Exemplo:
```python
return True, "OK"
```

### Mensagens
As mensagens devem mencionar o motivo. Os testes verificam palavras como `rodada`, `duplicada` e `terminou`.

---

### 7.5 `_consolidate_round_if_ready(self)`
Objetivo:
- verificar se todos os clientes já responderam e, se sim, consolidar a rodada.

### Primeira verificação obrigatória
Se ainda não chegaram todas as respostas:
```python
if len(self.received_updates) != self.num_clients:
    return
```

### Quando todos chegaram, faça nesta ordem
1. montar a lista de updates:
```python
updates = list(self.received_updates.values())
```
2. agregar com:
```python
self.global_params = aggregate_fedavg(updates)
```
3. avaliar o modelo global:
```python
loss, acc = evaluate_global_model(
    self.global_params,
    self.model_template,
    self.X_test_global,
    self.y_test_global,
)
```
4. calcular médias locais para impressão:
```python
avg_train_loss = sum(u["train_loss"] for u in updates) / len(updates)
avg_train_acc = sum(u["train_acc"] for u in updates) / len(updates)
```
5. imprimir logs da rodada;
6. limpar o buffer:
```python
self.received_updates = {}
```
7. avançar a rodada:
```python
self.current_round += 1
```
8. se a última rodada já terminou, sinalizar:
```python
if self.current_round > self.total_rounds:
    self.training_finished.set()
```

### Erro comum a evitar
Não incremente `self.current_round` antes de agregar e limpar o buffer da rodada atual.

---

### 7.6 `SubmitUpdate(self, request, context)`
Objetivo:
- receber a atualização de um cliente, validá-la, armazená-la e tentar consolidar a rodada.

### Ordem exata sugerida
1. validar:
```python
accepted, message = self._validate_update(request)
```
2. se inválida, retornar imediatamente:
```python
return federated_pb2.UpdateAck(
    accepted=False,
    message=message,
    server_round=self.current_round,
)
```
3. converter o modelo recebido:
```python
params = self._proto_to_params(request.model)
```
4. armazenar em `self.received_updates[request.cid]` um dicionário com:
- `params`
- `num_examples`
- `train_loss`
- `train_acc`
5. imprimir um log curto da rodada e do cliente;
6. chamar:
```python
self._consolidate_round_if_ready()
```
7. retornar `UpdateAck(accepted=True, ...)`.

### Estrutura sugerida para salvar o update
```python
self.received_updates[request.cid] = {
    "params": params,
    "num_examples": request.num_examples,
    "train_loss": request.train_loss,
    "train_acc": request.train_acc,
}
```

---

## 8. O que cada teste quer verificar

### `test_01_client_serialization.py`
Verifica se o cliente:
- serializa corretamente `coef` e `intercept`;
- reconstrói os arrays NumPy corretamente;
- monta o `ClientUpdate` com os campos certos.

### `test_02_client_loop.py`
Verifica se o cliente:
- consulta o servidor;
- treina localmente;
- envia uma atualização;
- respeita a lógica de rodada.

### `test_03_server_serialization_and_get.py`
Verifica se o servidor:
- serializa e desserializa parâmetros no mesmo formato do cliente;
- devolve o modelo global corretamente em `GetGlobalModel`;
- marca `done=True` quando o treinamento termina.

### `test_04_server_validation.py`
Verifica se o servidor:
- aceita update válido;
- rejeita update fora da rodada atual;
- rejeita update duplicado do mesmo cliente;
- rejeita update quando o treinamento já terminou.

### `test_05_server_submit_and_consolidate.py`
Verifica se o servidor:
- armazena updates recebidos;
- só avança a rodada quando todos os clientes responderem;
- agrega corretamente;
- limpa o buffer depois da consolidação;
- marca `training_finished` quando a última rodada termina.

---

## 9. Ordem recomendada para rodar os testes

Execute um por vez:

```bash
py -m unittest tests.test_01_client_serialization -v
py -m unittest tests.test_03_server_serialization_and_get -v
py -m unittest tests.test_04_server_validation -v
py -m unittest tests.test_05_server_submit_and_consolidate -v
py -m unittest tests.test_02_client_loop -v
```

### Por que essa ordem?
- `test_01` resolve as funções básicas do cliente;
- `test_03` resolve as funções básicas do servidor;
- `test_04` resolve a validação antes da consolidação;
- `test_05` resolve a lógica de rodada do servidor;
- `test_02` fica por último porque depende do cliente já mais completo.

> Sim, a ordem numérica dos arquivos não é a melhor ordem pedagógica. A ordem acima costuma funcionar melhor em aula.

---

## 10. Como executar todos os testes

Depois que as etapas forem concluídas, rode tudo:

```bash
python -m unittest discover -s tests -v
```

---

## 11. Execução manual depois que tudo estiver pronto

### Subir o servidor
```bash
python student_fl_server_grpc.py --host 127.0.0.1 --port 50051 --num-clients 3 --rounds 5
```

### Subir os clientes em outros terminais
```bash
python student_fl_client_grpc.py --cid 0 --server-address 127.0.0.1:50051 --num-clients 3
python student_fl_client_grpc.py --cid 1 --server-address 127.0.0.1:50051 --num-clients 3
python student_fl_client_grpc.py --cid 2 --server-address 127.0.0.1:50051 --num-clients 3
```

---

## 12. Dicas finais para os alunos

- implemente uma função por vez;
- rode o teste correspondente logo depois;
- quando um teste falhar, leia o nome do teste e a mensagem de erro com cuidado;
- mantenha o formato de serialização idêntico entre cliente e servidor;
- não tente resolver tudo ao mesmo tempo.

Se estiver em dúvida, pergunte primeiro:
- **quem chama essa função?**
- **o que ela deve devolver?**
- **qual teste depende dela?**

Essas três perguntas costumam ser suficientes para destravar a implementação.
