from utils import root_dir, ensure_dir
import tensorflow as tf
import numpy as np
from tensorflow.contrib import graph_editor as ge
from functools import partial
import pickle

input_shapes = {
    "vgg16": (1, 224, 224, 3),
    "vgg19": (1, 224, 224, 3),
    "inception_v1": (1, 224, 224, 3),
    "inception_v3": (1, 299, 299, 3),
    "resnet_v1_50": (1, 224, 224, 3),
    "resnet_v1_152": (1, 224, 224, 3),
    "resnet_v2_50": (1, 299, 299, 3),
    "resnet_v2_101": (1, 299, 299, 3),
    "resnet_v2_152": (1, 299, 299, 3),
    "resnet_v2_200": (1, 299, 299, 3),
    "mobilenet_v1_1.0": (1, 224, 224, 3),
    "mobilenet_v2_1.0_224": (1, 224, 224, 3),
    "inception_resnet_v2": (1, 299, 299, 3),
    "nasnet-a_large": (1, 331, 331, 3),
    "facenet": (1, 160, 160, 3),
    "rnn_lstm_gru_stacked": (1, 150),
}


def print_weights():
    for weight in tf.trainable_variables():
        print(weight.name)


def insert_masks(graph, masks):
    mask_placeholders = {}
    masked_weights = {}
    for weight in tf.trainable_variables():
        name = weight.name
        if name in masks:
            read_op = next(
                op for op in weight.op.outputs[0].consumers() if op.type == "Identity"
            )
            tensor = read_op.outputs[0]
            downstream_ops = tensor.consumers()
            mask_placeholder = tf.placeholder(dtype=tf.int32, shape=tensor.shape)
            masked_weight = tensor * tf.cast(mask_placeholder, dtype=tf.float32)
            for downstream_op in downstream_ops:
                downstream_op._update_input(
                    list(downstream_op.inputs).index(tensor), masked_weight
                )
            mask_placeholders[name] = mask_placeholder
            masked_weights[name] = masked_weight
    return mask_placeholders, masked_weights


class WarmStartHook(tf.train.SessionRunHook):
    def __init__(self, pretrained_model_dir, model_dir):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_dir = model_dir
        self.initialized = False

    def begin(self):
        if not self.initialized:
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if checkpoint is None:
                pretrained_checkpoint = tf.train.latest_checkpoint(
                    self.pretrained_model_dir
                )
                if pretrained_checkpoint is not None:
                    tf.train.warm_start(pretrained_checkpoint)
            self.initialized = True


class PruningHook(tf.train.SessionRunHook):
    def __init__(self, masks, mask_values):
        super().__init__()
        self.masks = masks
        self.mask_values = mask_values
        self.mask_placeholders = None
        self.masked_weights = None

    def begin(self):
        if len(self.mask_values) == 0:
            self.mask_values = {
                name: np.ones(tensor.shape, dtype=np.int)
                for name, tensor in self.mask_placeholders.items()
            }

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            fetches=self.masked_weights,
            feed_dict={
                placeholder: self.mask_values[name]
                for name, placeholder in self.mask_placeholders.items()
            },
        )

    def prune(self, masked_weights, mask_values):
        new_mask_values = mask_values
        return new_mask_values

    def after_run(self, run_context, run_values):
        masked_weights = run_values.results
        self.mask_values = self.prune(masked_weights, self.mask_values)


def model_fn(features, labels, mode, pretrained_model_dir, pruning_hook):
    checkpoint_file = tf.train.latest_checkpoint(pretrained_model_dir)
    tf.train.import_meta_graph(checkpoint_file + ".meta")
    input_tensor = tf.get_default_graph().get_tensor_by_name("input:0")
    for downstream_op in input_tensor.consumers():
        downstream_op._update_input(
            list(downstream_op.inputs).index(input_tensor), features
        )

    print_weights()
    mask_placeholders, masked_weights = insert_masks(
        tf.get_default_graph(), pruning_hook.masks
    )
    pruning_hook.mask_placeholders = mask_placeholders
    pruning_hook.masked_weights = masked_weights

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.sparse_softmax_cross_entropy(
            logits=tf.get_default_graph().get_tensor_by_name("MMdnn_Output:0"),
            labels=labels,
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op,
        )


def pruning(arch_name, masks):
    masks_file = ensure_dir(root_dir() / "masks" / (arch_name + ".pkl"))
    if masks_file.exists():
        with open(masks_file, "rb") as file:
            mask_values = pickle.load(file)
    else:
        mask_values = {}

    pretrained_model_dir = root_dir() / "downloads" / "model" / arch_name
    model_dir = root_dir() / "train" / arch_name

    warm_start_hook = WarmStartHook(pretrained_model_dir, model_dir)
    pruning_hook = PruningHook(masks, mask_values)

    estimator = tf.estimator.Estimator(
        model_fn=partial(
            model_fn,
            pruning_hook=pruning_hook,
            pretrained_model_dir=pretrained_model_dir,
        ),
        model_dir=model_dir,
    )
    input_fn = tf.estimator.inputs.numpy_input_fn(
        np.random.rand(*input_shapes[arch_name]).astype("f"),
        np.random.randint(1000, size=(1, 1)),
        batch_size=1,
        shuffle=False,
    )

    estimator.train(input_fn=input_fn, hooks=[warm_start_hook, pruning_hook])

    with open(masks_file, "wb") as file:
        pickle.dump(pruning_hook.mask_values, file)
    return


if __name__ == "__main__":
    pruning(
        "vgg16",
        masks=["vgg_16/conv1/conv1_1/weights:0", "vgg_16/conv5/conv5_3/weights:0"],
    )
