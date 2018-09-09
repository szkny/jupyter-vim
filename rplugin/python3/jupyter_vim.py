# =============================================================================
#    File: pythonx/jupyter_vim.py
# Created: 07/28/11 22:14:58
#  Author: Paul Ivanov (http://pirsquared.org)
#  Updated: [11/13/2017] Marijn van Vliet
#  Updated: [02/14/2018, 12:31] Bernie Roesler
#
# Description:
"""
Python code for ftplugin/python/jupyter.vim.
"""
# =============================================================================

from __future__ import print_function
import os
import re
import signal
import sys

import textwrap
from queue import Empty

_install_instructions = """You *must* install IPython into the Python that
your vim is linked against. If you are seeing this message, this usually means
either (1) installing IPython using the system Python that vim is using, or
(2) recompiling Vim against the Python where you already have IPython
installed. This is only a requirement to allow Vim to speak with an IPython
instance using IPython's own machinery. It does *not* mean that the IPython
instance with which you communicate via vim-ipython needs to be running the
same version of Python.
"""

try:
    import IPython
except ImportError:
    raise ImportError("Could not find kernel. " + _install_instructions)

try:
    import neovim
except ImportError:
    raise ImportError('vim module only available within vim!')


# ------------------------------------------------------------------------------
#        Read global configuration variables
# ------------------------------------------------------------------------------
is_py3 = sys.version_info[0] >= 3
if is_py3:
    unicode = str

prompt_in = 'In [{line:d}]: '
prompt_out = 'Out[{line:d}]: '


@neovim.plugin
class jupyter_vim(object):
    def __init__(self, nvim):
        self.nvim = nvim

        # define kernel manager, client and kernel pid
        self.kc = None
        self.pid = None
        self.send = None

        # ------------------------------------------------------------------------------
        #        Utilities
        # ------------------------------------------------------------------------------
        # Define wrapper for encoding
        # get around unicode problems when interfacing with vim
        self.vim_encoding = self.nvim.eval('&encoding') or 'utf-8'

    def vim_echom(self, arg, style="None"):
        """
        Report string `arg` using vim's echomessage command.

        Keyword args:
        style -- the vim highlighting style to use
        """
        try:
            self.nvim.command("echohl {}".format(style))
            self.nvim.command("echom \"{}\"".format(arg.replace('\"', '\\\"')))
            self.nvim.command("echohl None")
        except self.nvim.error:
            print("-- {}".format(arg))

    def check_connection(self):
        """Check that we have a client connected to the kernel."""
        return self.kc.hb_channel.is_beating() if self.kc else False

    def vim2py_str(self, var):
        """Convert to proper encoding."""
        if is_py3 and isinstance(var, bytes):
            var = str(var, self.vim_encoding)
        elif not is_py3 and isinstance(var, str):
            var = unicode(var, self.vim_encoding)
        return var

    # Taken from jedi-vim/pythonx/jedi_vim.py
    # <https://github.com/davidhalter/jedi-vim>

    def set_pid(self):
        """Explicitly ask the ipython kernel for its pid."""
        the_pid = -1
        code = 'import os; _pid = os.getpid()'
        msg_id = self.send(code, silent=True, user_expressions={'_pid': '_pid'})

        # wait to get message back from kernel
        try:
            reply = self.get_reply_msg(msg_id)
        except Empty:
            self.vim_echom("no reply from IPython kernel", "WarningMsg")
            return -1

        try:
            the_pid = int(reply['content']['user_expressions']
                          ['_pid']['data']['text/plain'])
        except KeyError:
            self.vim_echom("Could not get PID information, kernel not running Python?")

        return the_pid

    def is_cell_separator(self, line):
        """ Determine whether a given line is a cell separator """
        # TODO allow users to define their own cell separators
        cell_sep = ('##', '#%%', '# %%', '# <codecell>')
        return line.startswith(cell_sep)

    def strip_color_escapes(self, s):
        """Remove ANSI color escape sequences from a string."""
        strip = re.compile(r'\x1B\[([0-9]{1,2}(;[0-9]{1,2})*)?[mK]')
        return strip.sub('', s)

    # ------------------------------------------------------------------------------
    #        Major Function Definitions:
    # ------------------------------------------------------------------------------

    @neovim.command("TestCommand", range='', nargs=0)
    def connect_to_kernel(self):
        """Create kernel manager from existing connection file."""
        from jupyter_client import KernelManager, find_connection_file

        # Test if connection is alive
        connected = self.check_connection()
        attempt = 0
        max_attempts = 3
        while not connected and attempt < max_attempts:
            attempt += 1
            try:
                cfile = find_connection_file()  # default filename='kernel-*.json'
            except IOError:
                self.vim_echom("kernel connection attempt {:d} failed - no kernel file"
                          .format(attempt), style="Error")
                continue

            # Create the kernel manager and connect a client
            # See: <http://jupyter-client.readthedocs.io/en/stable/api/client.html>
            km = KernelManager(connection_file=cfile)
            km.load_connection_file()
            self.kc = km.client()
            self.kc.start_channels()

            # Alias execute function
            def _send(msg, **kwargs):
                """Send a message to the kernel client."""
                # Include dedent of msg so we don't get odd indentation errors.
                return self.kc.execute(textwrap.dedent(msg), **kwargs)
            self.send = _send

            # Ping the kernel
            self.kc.kernel_info()
            try:
                self.reply = self.kc.get_shell_msg(timeout=1)
            except Empty:
                continue
            else:
                connected = True

        if connected:
            # Send command so that monitor knows vim is commected
            # send('"_vim_client"', store_history=False)
            self.pid = self.set_pid()  # Ask kernel for its PID
            self.vim_echom('kernel connection successful! pid = {}'.format(self.pid),
                      style='Question')
        else:
            self.kc.stop_channels()
            self.vim_echom('kernel connection attempt timed out', style='Error')

    def disconnect_from_kernel(self):
        """Disconnect kernel client."""
        self.kc.stop_channels()
        self.vim_echom("Client disconnected from kernel with pid = {}".format(self.pid))

    def update_console_msgs(self):
        """Grab pending messages and place them inside the vim console monitor."""
        # Save which window we're in
        cur_win = self.nvim.eval('win_getid()')

        # Open the ipython terminal in vim, and move cursor to it
        is_console_open = self.nvim.eval('jupyter#OpenJupyterTerm()')
        if not is_console_open:
            self.vim_echom('__jupyter_term__ failed to open!', 'Error')
            return

        # Append the I/O to the console buffer
        io_pub = self.handle_messages()
        b = self.nvim.current.buffer
        for msg in io_pub:
            b.append([PythonToVimStr(l) for l in msg.splitlines()])
        self.nvim.command('normal! G')

        # Move cursor back to original window
        self.nvim.command(':call win_gotoid({})'.format(cur_win))

    def handle_messages(self):
        """
        Message handler for Jupyter protocol.

        Takes all messages on the I/O Public channel, including stdout, stderr,
        etc. and returns a list of the formatted strings of their content.

        See also: <http://jupyter-client.readthedocs.io/en/stable/messaging.html>
        """
        io_pub = []
        msgs = self.kc.iopub_channel.get_msgs(block=False)
        for msg in msgs:
            s = ''
            if 'msg_type' not in msg['header']:
                continue
            msg_type = msg['header']['msg_type']
            if msg_type == 'status':
                continue
            elif msg_type == 'stream':
                # TODO: alllow for distinguishing between stdout and stderr (using
                # custom syntax markers in the vim-ipython buffer perhaps), or by
                # also echoing the message to the status bar
                s = self.strip_color_escapes(msg['content']['text'])
            elif msg_type == 'display_data':
                s += msg['content']['data']['text/plain']
            elif msg_type == 'pyin' or msg_type == 'execute_input':
                line_number = msg['content'].get('execution_count', 0)
                prompt = prompt_in.format(line=line_number)
                s = prompt
                # add continuation line, if necessary
                dots = (' ' * (len(prompt.rstrip()) - 4)) + '...: '
                s += msg['content']['code'].rstrip().replace('\n', '\n' + dots)
            elif msg_type == 'pyout' or msg_type == 'execute_result':
                s = prompt_out.format(line=msg['content']['execution_count'])
                s += msg['content']['data']['text/plain']
            elif msg_type == 'pyerr' or msg_type == 'error':
                s = "\n".join(
                    map(self.strip_color_escapes, msg['content']['traceback']))
            elif msg_type == 'input_request':
                self.vim_echom('python input not supported in vim.', 'Error')
                continue  # unsure what to do here... maybe just return False?
            else:
                self.vim_echom("Message type {} unrecognized!".format(msg_type))
                continue

            # List all messages
            io_pub.append(s)

        return io_pub

    # ------------------------------------------------------------------------------
    #        Communicate with Kernel
    # ------------------------------------------------------------------------------

    def get_reply_msg(self, msg_id):
        """Get kernel reply from sent client message with msg_id."""
        # TODO handle 'is_complete' requests?
        # <http://jupyter-client.readthedocs.io/en/stable/messaging.html#code-completeness>
        while True:
            try:
                # TODO try block=False
                m = self.kc.get_shell_msg(block=False, timeout=1)
            except Empty:
                continue
            if m['parent_header']['msg_id'] == msg_id:
                return m

    def print_prompt(self, prompt, msg_id=None):
        """Print In[] or In[56] style messages on Vim's display line."""
        if msg_id:
            # wait to get message back from kernel
            try:
                self.reply = self.get_reply_msg(msg_id)
                count = self.reply['content']['execution_count']
                self.vim_echom("In[{:d}]: {:s}".format(count, prompt))
            except Empty:
                # if the kernel is waiting for input it's normal to get no reply
                if not self.kc.stdin_channel.msg_ready():
                    self.vim_echom("In[]: {} (no reply from IPython kernel)"
                              .format(prompt))
        else:
            self.vim_echom("In[]: {}".format(prompt))

    def run_command(self, cmd):
        """Send a single command to the kernel."""
        if not self.check_connection():
            self.vim_echom('WARNING: Not connected to IPython!', 'WarningMsg')
            return
        monitor_console = bool(int(self.nvim.vars.get('jupyter_monitor_console', 0)))
        verbose = bool(int(self.nvim.vars.get('jupyter_verbose', 0)))
        (prompt, msg_id) = self.send(cmd)
        if monitor_console:
            self.update_console_msgs()
        if verbose:
            self.print_prompt(prompt, msg_id=msg_id)
        return (cmd, msg_id)

    def run_file(self, flags='', filename=''):
        """Run a given python file using ipython's %run magic."""
        if not self.check_connection():
            self.vim_echom('WARNING: Not connected to IPython!', 'WarningMsg')
            return
        monitor_console = bool(int(self.nvim.vars.get('jupyter_monitor_console', 0)))
        verbose = bool(int(self.nvim.vars.get('jupyter_verbose', 0)))
        ext = os.path.splitext(filename)[-1][1:]
        if ext in ('pxd', 'pxi', 'pyx', 'pyxbld'):
            cmd = ' '.join(filter(None, (
                '%run_cython',
                self.vim2py_str(self.nvim.vars.get('cython_run_flags', '')),
                repr(filename))))
        else:
            b = self.nvim.current.buffer
            cmd = '%run {} {}'.format((flags or self.vim2py_str(b.vars['ipython_run_flags'])),
                                      repr(filename))
        (prompt, msg_id) = self.send(cmd)
        if monitor_console:
            self.update_console_msgs()
        if verbose:
            self.print_prompt(prompt, msg_id=msg_id)
        return (cmd, msg_id)

    def send_range(self):
        """Send a range of lines from the current vim buffer to the kernel."""
        if not self.check_connection():
            self.vim_echom('WARNING: Not connected to IPython!', 'WarningMsg')
            return
        monitor_console = bool(int(self.nvim.vars.get('jupyter_monitor_console', 0)))
        verbose = bool(int(self.nvim.vars.get('jupyter_verbose', 0)))
        r = self.nvim.current.range
        lines = "\n".join(self.nvim.current.buffer[r.start:r.end+1])
        msg_id = self.send(lines)
        prompt = "range %d-%d " % (r.start+1, r.end+1)
        if monitor_console:
            self.update_console_msgs()
        if verbose:
            self.print_prompt(prompt, msg_id=msg_id)
        return (prompt, msg_id)

    def run_cell(self):
        """Run all the code between two cell separators"""
        if not self.check_connection():
            self.vim_echom('WARNING: Not connected to IPython!', 'WarningMsg')
            return
        monitor_console = bool(int(self.nvim.vars.get('jupyter_monitor_console', 0)))
        verbose = bool(int(self.nvim.vars.get('jupyter_verbose', 0)))

        cur_buf = self.nvim.current.buffer
        (cur_line, cur_col) = self.nvim.current.window.cursor
        cur_line -= 1

        # Search upwards for cell separator
        upper_bound = cur_line
        while upper_bound > 0 and not self.is_cell_separator(cur_buf[upper_bound]):
            upper_bound -= 1

        # Skip past the first cell separator if it exists
        if self.is_cell_separator(cur_buf[upper_bound]):
            upper_bound += 1

        # Search downwards for cell separator
        lower_bound = min(upper_bound+1, len(cur_buf)-1)

        while lower_bound < len(cur_buf)-1 and \
                not self.is_cell_separator(cur_buf[lower_bound]):
            lower_bound += 1

        # Move before the last cell separator if it exists
        if self.is_cell_separator(cur_buf[lower_bound]):
            lower_bound -= 1

        # Make sure bounds are within buffer limits
        upper_bound = max(0, min(upper_bound, len(cur_buf)-1))
        lower_bound = max(0, min(lower_bound, len(cur_buf)-1))

        # Make sure of proper ordering of bounds
        lower_bound = max(upper_bound, lower_bound)

        # Execute cell
        lines = "\n".join(cur_buf[upper_bound:lower_bound+1])
        msg_id = self.send(lines)
        prompt = "execute lines {:d}-{:d} ".format(upper_bound+1, lower_bound+1)

        if monitor_console:
            self.update_console_msgs()
        if verbose:
            self.print_prompt(prompt, msg_id=msg_id)
        return (prompt, msg_id)

    def signal_kernel(self, sig=signal.SIGTERM):
        """
        Use kill command to send a signal to the remote kernel. This side steps the
        (non-functional) ipython interrupt mechanisms.
        Only works on posix.
        """
        try:
            os.kill(self.pid, int(sig))
            self.vim_echom("kill pid {p:d} with signal #{v:d}, {n:s}"
                      .format(p=self.pid, v=sig.value, n=sig.name), style='WarningMsg')
        except ProcessLookupError:
            self.vim_echom(("pid {p:d} does not exist! " +
                       "Kernel may have been terminated by outside process")
                      .format(p=self.pid), style='Error')
        except OSError as e:
            self.vim_echom("signal #{v:d}, {n:s} failed to kill pid {p:d}"
                      .format(v=sig.value, n=sig.name, p=self.pid), style='Error')
            raise e


class PythonToVimStr(unicode):
    """ Vim has a different string implementation of single quotes """
    __slots__ = []

    def __new__(cls, obj, encoding='UTF-8'):
        if not (is_py3 or isinstance(obj, unicode)):
            obj = unicode.__new__(cls, obj, encoding)

        # Vim cannot deal with zero bytes:
        obj = obj.replace('\0', '\\0')
        return unicode.__new__(cls, obj)

    def __repr__(self):
        # this is totally stupid and makes no sense but vim/python unicode
        # support is pretty bad. don't ask how I came up with this... It just
        # works...
        # It seems to be related to that bug: http://bugs.python.org/issue5876
        if unicode is str:
            s = self
        else:
            s = self.encode('UTF-8')
        return '"%s"' % s.replace('\\', '\\\\').replace('"', r'\"')
# ==============================================================================
# ==============================================================================
