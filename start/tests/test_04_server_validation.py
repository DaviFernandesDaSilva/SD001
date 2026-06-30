import types
import unittest
from unittest import mock

import numpy as np

from tests.utils import load_student_server_module


class TestServerValidation(unittest.TestCase):
    def setUp(self):
        self.server_module = load_student_server_module(module_name="student_server_test_validation")

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
            return self.server_module.FederatedLearningServicer(num_clients=2, total_rounds=2)

    def _make_update(self, cid=0, round_number=1):
        return self.server_module.federated_pb2.ClientUpdate(
            cid=cid,
            round=round_number,
            num_examples=4,
            train_loss=0.5,
            train_acc=0.75,
            model=self.server_module.federated_pb2.ModelParameters(
                coef_values=[0.0] * 12,
                coef_shape=[3, 4],
                intercept_values=[0.0, 0.0, 0.0],
            ),
        )

    def test_validate_accepts_fresh_update(self):
        servicer = self._make_servicer()
        accepted, message = servicer._validate_update(self._make_update(cid=0, round_number=1))
        self.assertTrue(accepted)
        self.assertEqual(message, "OK")

    def test_validate_rejects_if_training_has_ended(self):
        servicer = self._make_servicer()
        servicer.current_round = 3
        accepted, message = servicer._validate_update(self._make_update(cid=0, round_number=2))
        self.assertFalse(accepted)
        self.assertIn("terminou", message.lower())

    def test_validate_rejects_stale_round(self):
        servicer = self._make_servicer()
        accepted, message = servicer._validate_update(self._make_update(cid=0, round_number=99))
        self.assertFalse(accepted)
        self.assertIn("rodada", message.lower())

    def test_validate_rejects_duplicate_client_for_same_round(self):
        servicer = self._make_servicer()
        servicer.received_updates[0] = {
            "params": [np.zeros((3, 4)), np.zeros(3)],
            "num_examples": 4,
            "train_loss": 0.5,
            "train_acc": 0.75,
        }
        accepted, message = servicer._validate_update(self._make_update(cid=0, round_number=1))
        self.assertFalse(accepted)
        self.assertIn("duplicada", message.lower())


if __name__ == "__main__":
    unittest.main()
