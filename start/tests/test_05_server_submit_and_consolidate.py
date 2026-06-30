import types
import unittest
from unittest import mock

import numpy as np

from tests.utils import load_student_server_module


class TestServerSubmitAndConsolidate(unittest.TestCase):
    def setUp(self):
        self.server_module = load_student_server_module(module_name="student_server_test_submit")

    def _make_servicer(self):
        fake_model = types.SimpleNamespace(
            coef_=np.zeros((3, 4), dtype=np.float64),
            intercept_=np.zeros(3, dtype=np.float64),
        )
        with mock.patch.object(self.server_module, "create_model_template", return_value=fake_model), \
             mock.patch.object(
                 self.server_module,
                 "load_all_test_sets",
                 return_value=(np.zeros((2, 4), dtype=np.float64), np.array([0, 1])),
             ):
            return self.server_module.FederatedLearningServicer(num_clients=2, total_rounds=1)

    def _make_update(self, servicer, cid, round_number, weight_value):
        coef = np.full((3, 4), weight_value, dtype=np.float64)
        intercept = np.full(3, weight_value, dtype=np.float64)
        proto = servicer._params_to_proto([coef, intercept])
        return self.server_module.federated_pb2.ClientUpdate(
            cid=cid,
            round=round_number,
            num_examples=4,
            train_loss=0.5 + cid,
            train_acc=0.6 + cid,
            model=proto,
        )

    def test_submit_update_stores_update_and_consolidates_when_all_clients_arrive(self):
        """Dica: após a última atualização da rodada, o servidor deve agregar, limpar o buffer e avançar a rodada."""
        servicer = self._make_servicer()

        aggregated = [np.full((3, 4), 7.0), np.full(3, 9.0)]

        with mock.patch.object(self.server_module, "aggregate_fedavg", return_value=aggregated) as agg_mock, \
             mock.patch.object(self.server_module, "evaluate_global_model", return_value=(0.12, 0.95)):
            ack1 = servicer.SubmitUpdate(self._make_update(servicer, cid=0, round_number=1, weight_value=1.0), None)
            self.assertTrue(ack1.accepted)
            self.assertEqual(servicer.current_round, 1, "A rodada não deve avançar antes de todos os clientes responderem.")
            self.assertEqual(len(servicer.received_updates), 1)

            ack2 = servicer.SubmitUpdate(self._make_update(servicer, cid=1, round_number=1, weight_value=2.0), None)
            self.assertTrue(ack2.accepted)
            self.assertEqual(servicer.current_round, 2, "Depois da consolidação, o servidor deve avançar para a próxima rodada.")
            self.assertEqual(servicer.received_updates, {}, "Depois de consolidar, o buffer da rodada deve ser limpo.")
            self.assertTrue(servicer.training_finished.is_set(), "Como havia apenas uma rodada, o servidor deve sinalizar término.")
            self.assertEqual(agg_mock.call_count, 1)
            np.testing.assert_array_equal(servicer.global_params[0], aggregated[0])
            np.testing.assert_array_equal(servicer.global_params[1], aggregated[1])

    def test_submit_update_rejects_invalid_request(self):
        servicer = self._make_servicer()
        bad_request = self._make_update(servicer, cid=0, round_number=99, weight_value=1.0)

        ack = servicer.SubmitUpdate(bad_request, None)

        self.assertFalse(ack.accepted)
        self.assertIn("rodada", ack.message.lower())


if __name__ == "__main__":
    unittest.main()
