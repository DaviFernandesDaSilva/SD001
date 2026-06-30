import argparse
import time

import grpc
import numpy as np

import federated_pb2
import federated_pb2_grpc
from common_fl import create_model_template, load_client_partition, local_train


# Dica para os alunos:
# Este arquivo tem duas responsabilidades principais:
# 1) converter parâmetros do modelo entre protobuf <-> numpy
# 2) consultar o servidor, treinar localmente e enviar a atualização


def params_from_proto(proto_params):
    coef = np.array(proto_params.coef_values, dtype=np.float64)
    coef = coef.reshape(proto_params.coef_shape)

    intercept = np.array(proto_params.intercept_values, dtype=np.float64)

    return [coef, intercept]



def params_to_proto(params):
    coef, intercept = params

    return federated_pb2.ModelParameters(
        coef_values=coef.ravel().tolist(),
        coef_shape=list(coef.shape),
        intercept_values=intercept.ravel().tolist()
    )


def build_client_update(cid, round_number, train_result):
    model = params_to_proto(train_result["params"])

    return federated_pb2.ClientUpdate(
        cid=cid,
        round=round_number,
        num_examples=train_result["num_examples"],
        train_loss=train_result["train_loss"],
        train_acc=train_result["train_acc"],
        model=model
    )




def should_wait(global_round, completed_round):
    """
    Decide se o cliente deve esperar sem treinar.

    Ideia:
    - se a rodada do servidor for menor ou igual à última rodada já concluída por este cliente,
      não há nada novo para fazer ainda.
    """
    return global_round <= completed_round



def run_client(cid: int, server_address: str, num_clients: int, poll_interval: float, local_epochs: int):
    X_train, y_train, _, _ = load_client_partition(cid, num_clients)
    model_template = create_model_template(num_clients)

    completed_round = 0

    with grpc.insecure_channel(server_address) as channel:
        stub = federated_pb2_grpc.FederatedLearningStub(channel)

        while True:
            global_model = stub.GetGlobalModel(
                federated_pb2.ClientHello(cid=cid)
            )

            if global_model.done:
                print(f"Cliente {cid}: treinamento finalizado.")
                break

            if should_wait(global_model.round, completed_round):
                time.sleep(poll_interval)
                continue

            global_params = params_from_proto(global_model.model)

            train_result = local_train(
                X_train=X_train,
                y_train=y_train,
                global_params=global_params,
                model_template=model_template,
                local_epochs=local_epochs,
            )

            update = build_client_update(
                cid,
                global_model.round,
                train_result
            )

            try:
                ack = stub.SubmitUpdate(update)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print("Servidor finalizado. Encerrando cliente.")
                    break
                raise

            if ack.accepted:
                completed_round = global_model.round
                print(
                    f"Round {completed_round} | "
                    f"Loss: {train_result['train_loss']:.4f} | "
                    f"Acc: {train_result['train_acc']:.4f}"
                )
            else:
                print(ack.message)

            time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server-address", default="127.0.0.1:50051")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    args = parser.parse_args()

    run_client(
        cid=args.cid,
        server_address=args.server_address,
        num_clients=args.num_clients,
        poll_interval=args.poll_interval,
        local_epochs=args.local_epochs,
    )
