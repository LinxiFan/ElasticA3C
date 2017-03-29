"""
Tmux and subprocess
"""
import subprocess as pc
import signal

def new_tmux_cmd(session, name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return name, "tmux send-keys -t {}:{} '{}' Enter".format(session, name, cmd)


def _env_str(env):
    "Helper for env dict -> command line"
    return ' '.join(['{}={}'.format(k, env[k]) for k in env])


def nohup_str(cmd, 
          stdout_log=None, 
          stderr_log=None, 
          append_out=False,
          append_err=False,
          join_out_err=False,
          env=None):
    """
    Args:
      join_out_err: True: > stdout_log 2>&1 and ignore stderr_log
      env: dict of {'env_var': 'env_value'}
    
    Returns:
      a string of command wrapped in nohup
    """
    if stdout_log:
        stdout_log = ('>> ' if append_out else '> ') + stdout_log
    else:
        stdout_log = '> /dev/null' 
        
    if stderr_log:
        stderr_log = ('2>> ' if append_err else '2> ') + stderr_log
    else:
        if join_out_err:
            stderr_log = '2>&1'
        else:
            stderr_log = '2> /dev/null'
        
    # environment variables for this command
    if env is None:
        env = {}
    cmd = ('{} nohup {} {} {} &'.format(_env_str(env), 
                                        cmd, 
                                        stdout_log, 
                                        stderr_log).strip()
            + ' echo $!')
    return cmd


def nohup(cmd, 
          stdout_log=None, 
          stderr_log=None, 
          append_out=False,
          append_err=False,
          join_out_err=False,
          env=None,
          verbose=True, 
          dryrun=False):
    """
    Args:
      join_out_err: True: > stdout_log 2>&1 and ignore stderr_log
      env: dict of {'env_var': 'env_value'}
    """
    cmd = nohup_str(cmd,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    append_out=append_out,
                    append_err=append_err,
                    join_out_err=join_out_err,
                    env=env)
        
    if dryrun: print('Dry run:')
    # environment variables for this command
    if verbose:
        print(cmd)
    # don't actually run anything
    if dryrun: return
    # dollar-bang gets the PID of the last backgrounded process
    return pc.check_output(cmd, shell=True).strip()


def nohup_py(cmd, *args, **kwargs):
    return nohup('python -u {}'.format(cmd), *args, **kwargs)


def bash(cmd, env=None):
    "Exactly simulate os.system()"
    return pc.Popen('{} {}'.format(_env_str(env), cmd), shell=True).wait()


def kill(pid, signal='INT'):
    if pid is None:
        return
    if not isinstance(signal, str):
        signal = str(signal)
    if isinstance(pid, list) or isinstance(pid, tuple):
        for p in pid:
            kill(p)
    else:
        if not isinstance(pid, str):
            pid = str(pid)
        pc.call(['kill', '-'+signal, pid])
        
        
class SignalReceived(Exception):
    """
    Raised when any signal received
    .signum: value of the signal, symbolic
    .signame: name of the signal, string
    """
    signum = -1
    signame = ''


def register_signals(signames=None):
    """
    signames is a list of supported signals: SIGINT, SIGSTOP, etc.
    if None, default to register all supported signals
    """
    all_signames = filter(lambda s: (s.startswith('SIG')
                                     # not sure why ...
                                     and not s.startswith('SIGC')
                                     # OS cannot catch these two
                                     and s not in ['SIGKILL', 'SIGSTOP']
                                     and '_' not in s),
                          dir(signal))
    signum_to_name = {getattr(signal, sig): sig for sig in all_signames}

    if signames is None:
        signames = all_signames

    def _sig_handler(signum, frame):
        signame = signum_to_name[signum]
        exc = SignalReceived('{} [{}] received'.format(signame, signum))
        exc.signum = signum
        exc.signame = signame
        exc.frame = frame
        raise exc
            
    for sig in signames:
        signal.signal(getattr(signal, sig), _sig_handler)
