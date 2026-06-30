import unittest

import numpy as np

from tests.utils import load_student_client_module


class TestClientSerialization(unittest.TestCase):
    def setUp(self):
        self.client = load_student_client_module(module_name="student_client_test_serialization")

    def test_params_from_proto_rebuilds_numpy_arrays(self):
        """Dica: use coef_shape para dar reshape em coef_values."""
        proto = self.client.federated_pb2.ModelParameters(
            coef_values=[1, 2, 3, 4, 5, 6],
            coef_shape=[2, 3],
            intercept_values=[0.5, -0.5],
        )

        params = self.client.params_from_proto(proto)

        self.assertEqual(params[0].shape, (2, 3), "O coeficiente deve ser reconstruído com o shape original.")
        np.testing.assert_array_equal(params[0], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
        np.testing.assert_array_equal(params[1], np.array([0.5, -0.5], dtype=np.float64))

    def test_params_to_proto_flattens_coef_and_keeps_shape(self):
        """Dica: a matriz vai achatada para o protobuf, mas o shape precisa ser enviado junto."""
        params = [
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
            np.array([1.5, -1.5], dtype=np.float64),
        ]

        proto = self.client.params_to_proto(params)

        self.assertEqual(proto.coef_values, [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(proto.coef_shape, [2, 2])
        self.assertEqual(proto.intercept_values, [1.5, -1.5])

    def test_build_client_update_copies_metrics_and_serialized_model(self):
        """Dica: train_result vem de local_train e deve ser empacotado em ClientUpdate."""
        train_result = {
            "num_examples": 8,
            "train_loss": 0.12,
            "train_acc": 0.91,
            "params": [
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                np.array([0.1, 0.2], dtype=np.float64),
            ],
        }

        update = self.client.build_client_update(cid=7, round_number=3, train_result=train_result)

        self.assertEqual(update.cid, 7)
        self.assertEqual(update.round, 3)
        self.assertEqual(update.num_examples, 8)
        self.assertAlmostEqual(update.train_loss, 0.12)
        self.assertAlmostEqual(update.train_acc, 0.91)
        self.assertEqual(update.model.coef_shape, [2, 2])
        self.assertEqual(update.model.coef_values, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(update.model.intercept_values, [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
