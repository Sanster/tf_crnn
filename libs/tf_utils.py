import datetime
import os

import tensorflow as tf


def save_flags(flags, save_dir):
    """
    Save TF flags into file, use date as file name
    :param flags: tf.app.flags
    :param save_dir: dir to save flags file
    """
    filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, filename)
    print("Save flags to %s" % filepath)

    def format_flag(flag):
        ret = ""
        if flag.using_default_value:
            ret += "(default) "
        ret += ("%s: %s" % (flag.name, flag.value))
        return ret

    exclude_keys = ["help", "helpfull"]
    with open(filepath, mode="w", encoding="utf-8") as f:
        for k in flags:
            if k in exclude_keys:
                continue

            flag = flags[k]
            flag_str = format_flag(flag)
            f.write(flag_str + "\n")
