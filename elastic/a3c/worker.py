#!/usr/bin/env python
import cv2
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
import os
import universe
from elastic.a3c.A3C import A3C, POLICY, ELASTIC
from elastic.envs.vnc import *
from elastic.envs.video_segment_wrapper import *
from elastic.utils.io.filesys import *
from elastic.utils.image import *
import distutils.version

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SCRIPT_PATH = os.path.realpath(__file__)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def run(args, server):
    # configure logging
    args.log_dir = f_expand(args.log_dir)
    logging_dir = f_join(args.log_dir, 'log')
    video_dir = f_join(args.log_dir, 'video')
    info_dir = f_join(args.log_dir, 'info') # other diagnostics, such as screenshot
    f_mkdir(logging_dir)
    f_mkdir(video_dir)
    f_mkdir(info_dir)
    universe.configure_logging('{}/{:0>2}.txt'.format(logging_dir, args.task))
    
    if args.test:
        mode = 'test-' + args.test
        logger.info('TEST MODE: ' + ('stochastic' if args.test == 's' else 'deterministic'))
    else:
        mode = 'train'
        
    # create env
#     env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes)
#     env = create_atari_env(args.env_id, mode=mode, use_stack=(POLICY == 'cnn'))
#     env = wrap_record_video(env, video_dir=video_dir)
    env = gym.make(args.env_id)
    # env = wrap_video_segment(env, mode)
    env = wrap_deepmind(env, mode)
    env = wrap_diagnostics(env, mode)
    trainer = A3C(env, args.task, visualize=args.visualize, mode=mode)

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    variables_local = [v for v in tf.global_variables() if v.name.startswith("local")]
    # DEBUG
    for v in tf.global_variables():
        print(v.name, v.get_shape())
    print('='*80)
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    init_local_op = tf.variables_initializer(variables_local)
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    event_dir = f_join(args.log_dir, mode)

    event_suffix = '_{}'.format(args.task) if mode == 'train' else ''
    summary_writer = tf.summary.FileWriter(event_dir + event_suffix)

    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=event_dir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = 1000000000
    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        if ELASTIC:
            sess.run(init_local_op)
        sess.run(trainer.global_sync)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)


def cluster_spec(num_workers, num_eval, num_ps, port):
    cluster = {}

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers + num_eval):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def main(_):
    """
Setting up Tensorflow for data parallel work
"""
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", choices=['worker', 'ps'], help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default=os.path.expanduser("~/Train"), help='Log directory path')
    parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
    parser.add_argument('--port', type=int, default=15000, help='Cluster port starting point')
    parser.add_argument('-t', '--test', type=str, default=None, choices=['d', 's'],
                        help="Choices: d - deterministic, s - stochastic.")
    parser.add_argument('--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

    # Add visualisation argument
    parser.add_argument('--visualize', action='store_true',
                        help="visualize the gym environment by running env.render() between each timestep")

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 2, 1, args.port)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        # JF: the following can be replaced by server.join() 
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()
