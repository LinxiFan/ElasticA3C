# Elastic A3C

Run `pip install -e .` to install the `a3c` executable.

Command: 

```
usage: a3c [-h] [-n NUM_WORKERS] [--remotes REMOTES] [-e ENV] [-s SUFFIX]
           [-l LOG_DIR] [--dry-run] [-p PORT_OFFSET] [-r] [-t]
           [--cuda-devices CUDA_DEVICES]

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of workers
  --remotes REMOTES     The address of pre-existing VNC servers and rewarders
                        to use (e.g. -r vnc://localhost:5900+15900,vnc://local
                        host:5901+15901).
  -e ENV, --env ENV     Environment short-name
  -s SUFFIX, --suffix SUFFIX
                        Optional env suffix, will be appended to the names of
                        logdir and tmux window
  -l LOG_DIR, --log-dir LOG_DIR
                        Root log directory path
  --dry-run             Print out commands rather than executing them
  -p PORT_OFFSET, --port-offset PORT_OFFSET
                        Port offset (+10 x offset) for Tensorboard and
                        ClusterSpec
  -r, --resume          Do not delete the old save dir of parameters, restart
                        the training from scratch.
  -t, --test            run evaluation on the most recent checkpoint of the
                        trained agent.
  --cuda-devices CUDA_DEVICES
                        CUDA_VISIBLE_DEVICES string as environment variable.
```
