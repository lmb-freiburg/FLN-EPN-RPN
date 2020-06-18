import numpy as np
import os
import math
import tensorflow as tf

def tf_resample_hyps(hyps, coeff_x, coeff_y):
    resampled_hyps = []
    for h in hyps:
        x_center = h[:, 0:1, :, :] / coeff_x
        y_center = h[:, 1:2, :, :] / coeff_y
        width = h[:, 2:3, :, :] / coeff_x
        height = h[:, 3:4, :, :] / coeff_y
        resampled_hyps.append(tf.concat([x_center, y_center, width, height], axis=1))
    return resampled_hyps

def create_session():
    tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    return session

def optimistic_restore(session, save_file, ignore_vars=None, verbose=False, ignore_incompatible_shapes=False):
    def vprint(*args, **kwargs):
        if verbose: print(*args, flush=True, ** kwargs)

    if ignore_vars is None:
        ignore_vars = []

    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.dtype, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes and not var in ignore_vars])
    restore_vars = []

    nonfinite_values = False

    with tf.variable_scope('', reuse=True):
        for var_name, var_dtype, saved_var_name in var_names:
            curr_var = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if saved_var_name in var.name][0]
            var_shape = curr_var.get_shape().as_list()
            print(curr_var)
            if var_shape == saved_shapes[saved_var_name]:
                tmp = reader.get_tensor(saved_var_name)

                # check if there are nonfinite values in the tensor
                if not np.all(np.isfinite(tmp)):
                    nonfinite_values = True
                    print('{0} contains nonfinite values!'.format(saved_var_name), flush=True)

                if isinstance(tmp, np.ndarray):
                    saved_dtype = tf.as_dtype(tmp.dtype)
                else:
                    saved_dtype = tf.as_dtype(type(tmp))
                if not saved_dtype.is_compatible_with(var_dtype):
                    raise TypeError('types are not compatible for {0}: saved type {1}, variable type {2}.'.format(
                        saved_var_name, saved_dtype.name, var_dtype.name))

                vprint('restoring    ', saved_var_name)
                restore_vars.append(curr_var)
            else:
                vprint('not restoring', saved_var_name, 'incompatible shape:', var_shape, 'vs',
                       saved_shapes[saved_var_name])
                if not ignore_incompatible_shapes:
                    raise RuntimeError(
                        'failed to restore "{0}" because of incompatible shapes: var: {1} vs saved: {2} '.format(
                            saved_var_name, var_shape, saved_shapes[saved_var_name]))
    if nonfinite_values:
        raise RuntimeError('"{0}" contains nonfinite values!'.format(save_file))
    saver = tf.train.Saver(var_list=restore_vars, restore_sequentially=True)
    saver.restore(session, save_file)

def tf_full_conn(input, activation=None, **kwargs):

    k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
    k_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    b_initializer = tf.zeros_initializer

    num_output = kwargs.pop('num_output', False)
    name = kwargs.pop('name', 'conv_no_name')

    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(input)

    dense_out = tf.layers.dense(fc1,
                                int(num_output),
                                activation=activation,
                                kernel_initializer=k_initializer,
                                bias_initializer=b_initializer,
                                kernel_regularizer=k_regularizer,
                                trainable=True,
                                name=name)
    output = tf.reshape(dense_out, [dense_out.shape[0], dense_out.shape[1], 1, 1])
    return output

def tf_adjusted_sigmoid(X, min, max):
    tf.add_to_collection('log_scale_bound', max)
    const = lambda z: tf.fill(X.get_shape(), z)
    min = tf.to_float(min)
    max = tf.to_float(max)
    range = max - min
    x_scaled = tf.multiply(X, const(4.0 / range))
    sig = tf.sigmoid(x_scaled)
    sig_scaled = tf.multiply(sig, const(range))
    if min != 0:
        sig_scaled_shifted = tf.add(sig_scaled, const(min))
    else:
        sig_scaled_shifted = sig_scaled
    return sig_scaled_shifted

def tf_average_weighted_norm(x, w):  # axis=num_of_samples
    # x shape (batch,axis,w,h)
    # w shape (batch,axis,w,h)
    sum_w = tf.reduce_sum(w, axis=1)  # (batch,w,h)
    sum_w_inv = tf.pow(tf.add(sum_w, tf.fill(sum_w.get_shape(), 1e-6 / 2.0)), -1)  # (batch,w,h)

    x_weighted = tf.multiply(x, w)  # (batch,axis,w,h)
    x_weighted_sum = tf.reduce_sum(x_weighted, axis=1)  # (batch,w,h)
    result = tf.multiply(x_weighted_sum, sum_w_inv)  # (batch,w,h)
    result = tf.expand_dims(result, axis=1)  # (batch,1,w,h)
    return result

def tf_get_gaussian_mixture_model_from_samples_bbox(samples, assignments):
    # samples list of (n,4,1,1)
    # assignments list of (n,num_modes,1,1)
    # both list has the size of num of hypotheses (20) (output of sampling network)

    # Transform samples from c_x,c_y,w,h to tl_x,tl_y,br_x,br_y
    samples_transformed = [
        tf.concat([h[:, 0:2, :, :] - h[:, 2:4, :, :] / 2, h[:, 0:2, :, :] + h[:, 2:4, :, :] / 2], axis=1) for
        h in samples]

    num_of_modes = assignments[0].shape[1]
    expanded_samples = [tf.expand_dims(samples_transformed[i], axis=1) for i in
                        range(len(samples_transformed))]  # list of (n,1,4,1,1)
    samples_concat = tf.concat(expanded_samples, axis=1)  # (n,20,4,1,1)

    expanded_assignments = [tf.expand_dims(assignments[i], axis=1) for i in
                            range(len(assignments))]  # list of (n,1,num_modes,1,1)
    assignments_adjusted = tf.nn.softmax(tf.concat(expanded_assignments, axis=1),
                                          dim=2)  # (n,20,num_modes,1,1)


    mixture_weights = []
    means = []
    log_sigmas = []
    for k in range(num_of_modes):
        y_ik = assignments_adjusted[:, :, k, :, :]  # (n,20,1,1)
        w_k = tf.expand_dims(tf.reduce_mean(y_ik, axis=1), axis=1)  # (n,1,1,1)

        mu_k_tl_x = tf_average_weighted_norm(samples_concat[:, :, 0, :, :], y_ik)  # (n,1,1,1)
        mu_k_tl_y = tf_average_weighted_norm(samples_concat[:, :, 1, :, :], y_ik)
        mu_k_br_x = tf_average_weighted_norm(samples_concat[:, :, 2, :, :], y_ik)
        mu_k_br_y = tf_average_weighted_norm(samples_concat[:, :, 3, :, :], y_ik)
        mu_k = tf.concat([mu_k_tl_x, mu_k_tl_y, mu_k_br_x, mu_k_br_y], axis=1)  # (n,4,1,1)

        mu_k_repeated = tf.concat([tf.expand_dims(mu_k, axis=1) for i in range(len(samples))],
                                      axis=1)  # (n,20,4,1,1)
        diff = tf.subtract(samples_concat, mu_k_repeated)  # (n,20,4,1,1)
        diff2 = tf.pow(diff, 2)
        var_k_tl_x = tf_average_weighted_norm(diff2[:, :, 0, :, :], y_ik)  # (n,1,1,1)
        var_k_tl_y = tf_average_weighted_norm(diff2[:, :, 1, :, :], y_ik)
        var_k_br_x = tf_average_weighted_norm(diff2[:, :, 2, :, :], y_ik)
        var_k_br_y = tf_average_weighted_norm(diff2[:, :, 3, :, :], y_ik)
        var_k = tf.concat([var_k_tl_x, var_k_tl_y, var_k_br_x, var_k_br_y], axis=1)  # (n,4,1,1)
        sigma_k = tf.pow(var_k, 0.5)
        log_sigma_k = tf.log(sigma_k)  # (n,4,1,1)

        mixture_weights.append(w_k)  # (n,1,1,1)
        means.append(mu_k)
        log_sigmas.append(log_sigma_k)

    return means, log_sigmas, mixture_weights

def tf_assemble_gmm_parameters_samples(samples_means, assignments):
    # samples list of (n,4,1,1)
    # assignments list of (n,num_modes,1,1)
    # both list has the size of num of hypotheses (20) (output of sampling network)
    means, log_sigmas, mixture_weights = tf_get_gaussian_mixture_model_from_samples_bbox(samples_means, assignments)
    # means list of (n,4,1,1), log_sigmas list of (n,4,1,1), mixture_weights are list of (n,1,1,1)
    bounded_log_sigmas = [tf_adjusted_sigmoid(log_sigmas[i], -6, 6) for i in range(len(log_sigmas))]

    return means, bounded_log_sigmas, mixture_weights

def tf_get_mask(indices, width, height, fill_value=1.0):# shape of indices is 6
    indices = tf.to_int32(indices)
    tl_x = tf.math.maximum(tf.math.minimum(indices[0], width - 1), 0)
    tl_y = tf.math.maximum(tf.math.minimum(indices[1], height - 1), 0)
    bbox_width = tf.math.maximum(tf.math.minimum(indices[2], width - tl_x - 2), 1)
    bbox_height = tf.math.maximum(tf.math.minimum(indices[3], height - tl_y - 2), 1)
    ind_row = [tl_y, height - tl_y - bbox_height]
    ind_col = [tl_x, width - tl_x - bbox_width]
    padding = tf.stack([ind_row, ind_col])
    input = tf.ones([bbox_height, bbox_width]) * (fill_value + 1)
    padded = tf.expand_dims(tf.expand_dims(tf.pad(input, padding, "CONSTANT"), axis=0), axis=0)
    return padded

def tf_impose_hyps(bboxes, width, height):  # list of bbox has shape 1,4,1,1
    width = int(width)
    height = int(height)
    all_cordinates = []
    for bbox in bboxes:
        center_x = bbox[:, 0, 0, 0] / width  # shape is n
        center_y = bbox[:, 1, 0, 0] / height
        bbox_width = bbox[:, 2, 0, 0] / width
        bbox_height = bbox[:, 3, 0, 0] / height
        min_x = center_x - bbox_width / 2
        min_y = center_y - bbox_height / 2
        max_x = min_x + bbox_width
        max_y = min_y + bbox_height
        coordinates = tf.stack([min_y, min_x, max_y, max_x], axis=1)  # n,4
        coordinates = tf.expand_dims(coordinates, axis=1)  # n,1,4
        all_cordinates.append(coordinates)
    concat_coordinates = tf.concat(all_cordinates, axis=1)  # n,num,4
    image_masked = tf.transpose(tf.image.draw_bounding_boxes(
        tf.zeros((1, height, width, 1)), concat_coordinates), perm=[0, 3, 1, 2])
    return image_masked