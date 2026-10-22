"""CAGRA 本机/ADB 执行封装；默认只展示命令，调用方显式执行。"""
from __future__ import annotations

import ast
import os
from pathlib import Path, PurePosixPath
import shlex
import struct
import subprocess


def add_execution_arguments(parser):
    parser.add_argument('--target', choices=['linux', 'android'], default='linux')
    parser.add_argument('--binary-dir', help='本机或手机上的程序目录')
    parser.add_argument('--device', help='ADB 设备序列号')
    parser.add_argument('--device-root', default='/data/local/tmp/ann')


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def uint(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < 2147483647:
        raise ValueError(f'{name} 必须是 [{minimum}, 2147483646] 范围内的整数')
    return value


class Runtime:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.android = args.target == 'android'
        self.adb = ['adb'] + (['-s', args.device] if args.device else [])
        if self.android:
            self.device_root = self.path(args.device_root)
            self.results = self.device_root / 'results' / 'cagra'

    def path(self, value):
        if not self.android:
            return Path(value).resolve()
        text = str(value)
        p = PurePosixPath(text)
        if not p.is_absolute() or text.startswith('//') or '..' in p.parts or any(c in text for c in ('\\', '\n', '\r', '\x00')):
            raise ValueError('Android 必须使用不含 .. 的设备 POSIX 绝对路径：' + text)
        return p

    def binary_dir(self):
        if self.args.binary_dir:
            return self.path(self.args.binary_dir)
        return self.device_root / 'bin/cagra' if self.android else self.root / 'build/bin'

    def output(self, value):
        p = self.path(value) if self.android else (self.root / value).resolve()
        allowed = self.results if self.android else (self.root / 'results').resolve()
        if not within(p, allowed) or p == allowed:
            raise ValueError('查询输出前缀必须位于 ' + str(allowed))
        return p

    def file_info(self, path, length=16):
        if self.android:
            q = shlex.quote(str(path))
            prefix = f'test -f {q} && '
            size = int(subprocess.check_output(self.adb + ['shell', prefix + f'stat -c %s {q}'], stdin=subprocess.DEVNULL))
            header = subprocess.check_output(self.adb + ['exec-out', prefix + f'head -c {length} {q}'], stdin=subprocess.DEVNULL)
        else:
            size = path.stat().st_size
            with path.open('rb') as stream:
                header = stream.read(length)
        return size, header

    def vector_rows(self, path):
        size, header = self.file_info(path)
        if header[:8] == b'EGF32001' and len(header) == 16:
            count, dim = struct.unpack('<II', header[8:])
            valid = dim == 128 and size == 16 + count * dim * 4
        else:
            dim = struct.unpack('<I', header[:4])[0] if len(header) >= 4 else 0
            count = size // (129 * 4)
            valid = dim == 128 and size % (129 * 4) == 0
        if not valid:
            raise ValueError('需要完整的 128 维 fvecs 或 EGF32001 文件：' + str(path))
        return uint(count, '文件向量数', 1)

    def code_rows(self, path):
        # Read only the NPY header. No NumPy dependency for launchers and no float base access.
        size, header = self.file_info(path, 65548)
        if header[:6] != b'\x93NUMPY' or header[6:8] not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
            raise ValueError('无效的 codes.npy 文件头')
        offset = 10 if header[6] == 1 else 12
        length = int.from_bytes(header[8:offset], 'little')
        if length > 65536 or len(header) < offset + length:
            raise ValueError('codes.npy 文件头过长或不完整')
        info = ast.literal_eval(header[offset:offset + length].decode('utf-8'))
        if info.get('fortran_order') or not isinstance(info.get('shape'), tuple) or not info['shape']:
            raise ValueError('需要非空 C-order codes.npy')
        count = uint(info['shape'][0], '编码向量数', 1)
        if size <= offset + length:
            raise ValueError('codes.npy 缺少编码载荷')
        return count

    def check_build(self, data, output):
        if self.android:
            allowed = self.device_root / 'indexes/cagra'
            if not within(output, allowed) or output == allowed:
                raise ValueError('Android 构图输出必须是 ' + str(allowed) + ' 内的新目录')
            check = f'test -f {shlex.quote(str(data))} && test ! -e {shlex.quote(str(output))}'
            subprocess.run(self.adb + ['shell', check], check=True, stdin=subprocess.DEVNULL)
        else:
            if output.exists():
                raise ValueError('构图输出目录已存在，请选择新目录')
            if not data.is_file():
                raise ValueError('底库文件不存在')
            if not within(output, self.root) or output == self.root:
                raise ValueError('本机构图输出必须位于 CAGRA 内')

    def wrapped(self, command, threads):
        env = ['env', f'OMP_NUM_THREADS={threads}', 'OPENBLAS_NUM_THREADS=1', 'LC_ALL=C']
        if self.android:
            return self.adb + ['shell', shlex.join(env + list(map(str, command)))]
        return env + list(map(str, command))

    def execute(self, commands, output, threads, build=False):
        """Each stage records stdout and time -v; query output is a prefix, not a file."""
        q = shlex.quote
        lock = str(output) + '.lock'
        if build:
            log_paths = [(output / 'knn', output / 'build')[i] for i in range(len(commands))]
        else:
            log_paths = [output]
        if self.android:
            out = q(str(output))
            reserve = f'mkdir {out}' if build else f'mkdir {q(lock)}'
            check = f'test ! -e {out}' if build else ' && '.join(
                f'test ! -e {q(str(output) + suffix)}' for suffix in ('.json', '.labels', '.stdout', '.time'))
            bodies = []
            for command, log in zip(commands, log_paths):
                env = ['env', f'OMP_NUM_THREADS={threads}', 'OPENBLAS_NUM_THREADS=1', 'LC_ALL=C']
                timed = ['/system/bin/time', '-v'] + env + list(map(str, command))
                bodies.append(shlex.join(timed) + f' > {q(str(log)+".stdout")} 2> {q(str(log)+".time")}')
            body = ' && '.join(bodies)
            cleanup = '' if build else f"trap {q('rmdir ' + q(lock))} EXIT; "
            # Query checks run after reservation so a completed concurrent run
            # cannot have its logs truncated between the check and lock creation.
            checked_body = body if build else f'{check} && {body}'
            before = f'{check} && ' if build else ''
            shell = f'test -x /system/bin/time && {before}mkdir -p {q(str(output.parent))} && {reserve} && ( {cleanup}{checked_body} )'
            subprocess.run(self.adb + ['shell', shell], check=True, stdin=subprocess.DEVNULL)
        else:
            timer = Path('/usr/bin/time')
            if not timer.is_file():
                raise ValueError('请安装 time 软件包，以记录构图/查询时间和峰值内存')
            if build:
                output.mkdir(parents=True, exist_ok=False)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                Path(lock).mkdir()
            try:
                if not build:
                    for suffix in ('.json', '.labels', '.stdout', '.time'):
                        if Path(str(output) + suffix).exists():
                            raise ValueError('输出已存在：' + str(output) + suffix)
                environment = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',
                                   PYTHONDONTWRITEBYTECODE='1', LC_ALL='C')
                for command, log in zip(commands, log_paths):
                    with Path(str(log)+'.stdout').open('x') as stdout, Path(str(log)+'.time').open('x') as stderr:
                        subprocess.run([str(timer), '-v'] + list(map(str, command)), env=environment,
                                       stdout=stdout, stderr=stderr, check=True, stdin=subprocess.DEVNULL)
            finally:
                if not build:
                    Path(lock).rmdir()
