import os
import h5py
import json


# def get_all_scores(y_true, y_pred, y_fields, weights=None):
#     def get_column_score(arr_true, arr_pred, w):
#         r2, rmse, mae, r_rmse, sr_rmse, cor_value = [np.zeros(arr_true.shape[0]) for _ in range(6)]
#         for i in range(arr_true.shape[0]):
#             arr_true_i = arr_true[i, w[i, :]]
#             arr_pred_i = arr_pred[i, w[i, :]]
#
#             r2[i] = r2_score(arr_true_i, arr_pred_i)
#             rmse[i] = np.sqrt(mse(arr_true_i, arr_pred_i))
#             mae[i] = np.mean(abs((arr_true_i - arr_pred_i)))
#             r_rmse[i] = rmse[i] / (arr_true_i.max() - arr_true_i.min())
#             sr_rmse[i] = rmse[i] / (0.5 * (arr_true_i.max() - arr_true_i.min() + arr_pred_i.max() - arr_pred_i.min()))
#             cor_value[i] = pearsonr(arr_true_i, arr_pred_i)[0]
#
#         locs = np.where(w.ravel())[0]
#         r2_all = r2_score(arr_true.ravel()[locs], arr_pred.ravel()[locs])
#         r2_all = np.full(r2.shape, r2_all)
#         return {'r2': r2, 'r2_all': r2_all, 'rmse': rmse, 'mae': mae, 'r_rmse': r_rmse, 'sr_rmse': sr_rmse,
#                 'cor_value': cor_value}
#
#     scores = []
#     weights = {} if weights is None else weights
#     for output_name, fields in y_fields.items():
#         for col, field in enumerate(fields):
#             y_true_one_field = y_true[output_name][:, :, col]
#             y_pred_one_field = y_pred[output_name][:, :, col]
#             try:
#                 weight_one_field = weights[output_name][:, :, col] == 1.
#             except KeyError:
#                 weight_one_field = np.full(y_true_one_field.shape, True)
#                 logging.warning("Use default all true value for {}".format(output_name))
#             score_one_field = {'output': output_name, 'field': field}
#             score_one_field.update(get_column_score(y_true_one_field, y_pred_one_field, weight_one_field))
#             scores.append(score_one_field)
#     return scores

if __name__ == '__main__':
    result_dir = './results/0107_KAM/IMU+OP'
    with h5py.File(os.path.join(result_dir, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])
    print(1)
