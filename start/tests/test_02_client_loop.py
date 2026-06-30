import types
import unittest
from unittest import mock

import numpy as np

from tests.utils import DummyContextManager, load_student_client_module


class FakeStub:
    def __init__(self, client_module):
        self.client_module = client_module
        self.updates = []
        self.get_calls = 0

    def GetGlobalModel(self, request):
        self.get_calls += 1
        # Primeira chamada: há uma rodada nova para processar.
        if self.get_calls == 1:
            return self.client_module.federated_pb2.GlobalModel(
                round=1,
                total_rounds=1,
                done=False,
                message="rodada 1",
                model=self.client_module.federated_pb2.ModelParameters(
                    coef_values=[0.0] * 12,
                    coef_shape=[3, 4],
                    intercept_values=[0.0, 0.0, 0.0],
                ),
            )

        # Segunda chamada: servidor já encerrou.
        return self.client_module.federated_pb2.GlobalModel(
            round=2,
            total_rounds=1,
            done=True,
            message="fim",
            model=self.client_module.federated_pb2.ModelParameters(
                coef_values=[0.0] * 12,
                coef_shape=[3, 4],
                intercept_values=[0.0, 0.0, 0.0],
            ),
        )

    def SubmitUpdate(self, update):
        self.updates.append(update)
        return self.client_module.federated_pb2.UpdateAck(
            accepted=True,
            message="ok",
            server_round=2,
        )


class TestClientLoop(unittest.TestCase):
    def setUp(self):
        self.client = load_student_client_module(module_name="student_client_test_loop")

    def test_should_wait_is_true_only_for_old_or_equal_rounds(self):
        self.assertTrue(self.client.should_wait(global_round=1, completed_round=1))
        self.assertTrue(self.client.should_wait(global_round=1, completed_round=2))
        self.assertFalse(self.client.should_wait(global_round=2, completed_round=1))

    def test_run_client_processes_one_round_and_sends_update(self):
        """Dica: o cliente deve consultar o modelo, treinar, enviar a atualização e encerrar quando done=True."""
        fake_stub = FakeStub(self.client)

        with mock.patch.object(self.client, "load_client_partition", return_value=(
            np.zeros((4, 4), dtype=np.float64),
            np.array([0, 1, 0, 1]),
            np.zeros((1, 4), dtype=np.float64),
            np.array([0]),
        )), mock.patch.object(
            self.client,
            "create_model_template",
            return_value=types.SimpleNamespace(
                coef_=np.zeros((3, 4), dtype=np.float64),
                intercept_=np.zeros(3, dtype=np.float64),
                classes_=np.array([0, 1, 2]),
            ),
        ), mock.patch.object(
            self.client,
            "local_train",
            return_value={
                "params": [np.ones((3, 4), dtype=np.float64), np.ones(3, dtype=np.float64)],
                "num_examples": 4,
                "train_loss": 0.4,
                "train_acc": 0.75,
            },
        ), mock.patch.object(
            self.client.federated_pb2_grpc,
            "FederatedLearningStub",
            return_value=fake_stub,
        ), mock.patch.object(
            self.client.grpc,
            "insecure_channel",
            return_value=DummyContextManager(object()),
        ), mock.patch.object(self.client.time, "sleep", return_value=None):
            self.client.run_client(
                cid=0,
                server_address="127.0.0.1:50051",
                num_clients=3,
                poll_interval=0.0,
                local_epochs=1,
            )

        self.assertEqual(len(fake_stub.updates), 1, "O cliente deve enviar exatamente uma atualização nesta simulação.")
        sent_update = fake_stub.updates[0]
        self.assertEqual(sent_update.cid, 0)
        self.assertEqual(sent_update.round, 1)
        self.assertEqual(sent_update.num_examples, 4)
        self.assertAlmostEqual(sent_update.train_loss, 0.4)
        self.assertAlmostEqual(sent_update.train_acc, 0.75)


if __name__ == "__main__":
    unittest.main()
