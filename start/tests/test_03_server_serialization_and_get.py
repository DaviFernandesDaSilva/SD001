import types
import unittest
from unittest import mock

import numpy as np

from tests.utils import load_student_server_module


class TestServerSerializationAndGet(unittest.TestCase):
    def setUp(self):
        self.server_module = load_student_server_module(module_name="student_server_test_get")

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
            return self.server_module.FederatedLearningServicer(num_clients=2, total_rounds=3)

    def test_params_to_proto_serializes_shape_and_values(self):
        """Dica: servidor e cliente devem usar exatamente o mesmo formato de serialização."""
        servicer = self._make_servicer()
        params = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([5.0, 6.0], dtype=np.float64),
        ]

        proto = servicer._params_to_proto(params)

        self.assertEqual(proto.coef_values, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(proto.coef_shape, [2, 2])
        self.assertEqual(proto.intercept_values, [5.0, 6.0])

    def test_proto_to_params_rebuilds_numpy_arrays(self):
        servicer = self._make_servicer()
        proto = self.server_module.federated_pb2.ModelParameters(
            coef_values=[1, 2, 3, 4, 5, 6],
            coef_shape=[3, 2],
            intercept_values=[7, 8, 9],
        )

        params = servicer._proto_to_params(proto)

        np.testing.assert_array_equal(params[0], np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64))
        np.testing.assert_array_equal(params[1], np.array([7, 8, 9], dtype=np.float64))

    def test_get_global_model_returns_current_round(self):
        """Dica: done deve ser False enquanto current_round <= total_rounds."""
        servicer = self._make_servicer()
        response = servicer.GetGlobalModel(
            self.server_module.federated_pb2.ClientHello(cid=0),
            context=None,
        )

        self.assertEqual(response.round, 1)
        self.assertEqual(response.total_rounds, 3)
        self.assertFalse(response.done)
        self.assertIsNotNone(response.model)

    def test_get_global_model_marks_done_after_last_round(self):
        servicer = self._make_servicer()
        servicer.current_round = 4

        response = servicer.GetGlobalModel(
            self.server_module.federated_pb2.ClientHello(cid=0),
            context=None,
        )

        self.assertTrue(response.done, "Quando current_round > total_rounds, o servidor deve sinalizar fim do treinamento.")
        self.assertEqual(response.round, 4)


if __name__ == "__main__":
    unittest.main()
