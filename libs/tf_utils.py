import tensorflow as tf


def add_scalar_summary(writer, tag, val, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
    writer.add_summary(summary, step)
