class Logger(object):
    def __init__(self, log_dir):
        self.write = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.write.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.write.flush()
        # summary = tf.summary(value=[tf.summary.Value(tag=tag, simple_value=value)])
        # self.write.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        with self.write.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step)
            self.write.flush()
        # summary =tf.summary(value=[tf.summary.Value(tag=tag, sample_value=value) for tag, value in tag_value_pairs])
        # self.write.add_summary(summary, step)
