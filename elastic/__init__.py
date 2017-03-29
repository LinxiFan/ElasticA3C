"""
Stanford RL Group
"""
import sys

def govnc_guard():
    """
    Workaround for OpenAI Gym: go_vncdriver must be imported before TF

    Example:
    
      Put this at the top of a module::

        from elastic import govnc_guard
        if govnc_guard():
            import go_vncdriver
        import tensorflow
    """
    return 'go_vncdriver' not in sys.modules and 'tensorflow' not in sys.modules


# exec(safe_tf_import)
safe_tf_import = """
import sys
if 'go_vncdriver' not in sys.modules and 'tensorflow' not in sys.modules:
    import go_vncdriver
import tensorflow as tf
"""
