"""Local/ADB execution for this index's public launchers (Python 3.8+)."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shlex
import struct
import subprocess


def add_execution_arguments(parser):
    parser.add_argument('--target', choices=['linux', 'android'], default='linux')
    parser.add_argument('--binary-dir', help='Local binary directory, or absolute device directory for Android')
    parser.add_argument('--device', help='ADB serial; otherwise ADB requires exactly one device')
    parser.add_argument('--device-root', default='/data/local/tmp/ann', help='Android runtime root')


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class Runtime:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.android = args.target == 'android'
        self.adb = ['adb'] + (['-s', args.device] if args.device else [])
        if self.android:
            self.device_root = self.path(args.device_root)
            self.results = self.device_root / 'results' / root.name.lower()

    def path(self, value):
        if not self.android:
            return Path(value).resolve()
        text = str(value)
        p = PurePosixPath(text)
        if not p.is_absolute() or text.startswith('//') or '..' in p.parts or '\\' in text or '\n' in text or '\r' in text:
            raise ValueError('Android paths must be absolute POSIX paths without ..: ' + text)
        return p

    def binary_dir(self):
        if self.args.binary_dir:
            return self.path(self.args.binary_dir)
        if self.android:
            return self.device_root / 'bin' / self.root.name.lower()
        # Preserve the GitHub default; platform-separated builds can be selected explicitly.
        return self.root / 'build/bin'

    def output(self, value):
        if self.android:
            p = self.path(value)
            if not within(p, self.results) or p == self.results:
                raise ValueError('Android query output must be inside ' + str(self.results))
        else:
            p = (self.root / value).resolve()
            if not within(p, (self.root / 'results').resolve()) or p == (self.root / 'results').resolve():
                raise ValueError("Output must be inside this index's results directory")
        return p

    def query_rows(self, path, index):
        if self.android:
            # Inspect the device file, not a same-named file on the SSH host.
            q = shlex.quote(str(path))
            prefix = f'test -f {q} && '
            size = int(subprocess.check_output(self.adb + ['shell', prefix + f'stat -c %s {q}'], stdin=subprocess.DEVNULL))
            header = subprocess.check_output(self.adb + ['exec-out', prefix + f'head -c 8 {q}'], stdin=subprocess.DEVNULL)
        else:
            size = path.stat().st_size
            with path.open('rb') as stream:
                header = stream.read(8)
        if index == 'VAMANA':
            if len(header) != 8:
                raise ValueError('Truncated query fbin header')
            count, dimension = struct.unpack('<II', header)
            valid = dimension == 128 and size == 8 + count * dimension * 4
        else:
            dimension = struct.unpack('<I', header[:4])[0] if len(header) >= 4 else 0
            count = size // (129 * 4)
            valid = dimension == 128 and size % (129 * 4) == 0
        if not valid or not 0 < count <= 0xffffffff:
            raise ValueError('Expected a nonempty 128-dimensional query file with complete records')
        return count

    def check_build(self, data, output):
        if self.android:
            allowed = self.device_root / 'indexes' / self.root.name.lower()
            if not within(output, allowed) or output == allowed:
                raise ValueError('Android build output must be a new directory inside ' + str(allowed))
            # The same checks run atomically with mkdir immediately before execution.
            check = f'test -f {shlex.quote(str(data))} && test ! -e {shlex.quote(str(output))}'
            subprocess.run(self.adb + ['shell', check], check=True, stdin=subprocess.DEVNULL)
        else:
            if output.exists():
                raise ValueError('Output directory already exists; choose a new directory')
            if not data.is_file():
                raise ValueError('Input data file does not exist')

    def wrapped(self, command, threads):
        environment = ['env', 'OMP_NUM_THREADS=' + str(threads), 'OPENBLAS_NUM_THREADS=1', 'LC_ALL=C']
        if self.android:
            return self.adb + ['shell', shlex.join(environment + list(map(str, command)))]
        return list(map(str, command))

    def execute(self, commands, output, threads, build=False):
        if self.android:
            out = shlex.quote(str(output))
            parent = shlex.quote(str(output.parent))
            # Noclobber reserves query output; exclusive mkdir reserves graph output.
            reserve = f'mkdir {out}' if build else f'(set -C; : > {out})'
            body = ' && '.join(shlex.join(['env', 'OMP_NUM_THREADS=' + str(threads),
                                         'OPENBLAS_NUM_THREADS=1', 'LC_ALL=C'] + list(map(str, c))) for c in commands)
            subprocess.run(self.adb + ['shell', f'test ! -e {out} && test ! -L {out} && mkdir -p {parent} && {reserve} && {body}'],
                           check=True, stdin=subprocess.DEVNULL)
        else:
            if build:
                output.mkdir(parents=True, exist_ok=False)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive reservation prevents accidental reuse of an existing result.
                with output.open('xb'):
                    pass
            environment = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',
                               PYTHONDONTWRITEBYTECODE='1', LC_ALL='C')
            for command in commands:
                subprocess.run(list(map(str, command)), env=environment, check=True, stdin=subprocess.DEVNULL)
