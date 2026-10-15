"""Client for the project object store, over HTTP or on a local mount of the store volume."""

import os
import shutil
import subprocess
import time

import requests

URL = os.environ.get('TRI_STORE_URL', '')
TOKEN_PATH = '/content/tri/.store_token'
LOCAL_ROOT = os.environ.get('TRI_STORE_DIR', '')  # the store volume, mounted on the store's own compute
TRANSFER_TRIES = 4
USER_AGENT = {'User-Agent': 'trishuli-hackathon/0.1 (research)'}


def _token():
    token = os.environ.get('TRI_STORE_TOKEN')
    if not token and os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as f:
            token = f.read().strip()
    return token


def has_token():
    return bool(_token())


def _headers():
    return {'x-tri-token': _token()}


def _is_local():
    return bool(LOCAL_ROOT) and os.path.isdir(LOCAL_ROOT)


def _copy_via_part_file(source_path, target_path):
    """Copy so that target_path never exists half-written."""
    shutil.copyfile(source_path, target_path + '.part')
    os.replace(target_path + '.part', target_path)


def _put_local(local_path, remote_path):
    if '..' in remote_path:
        raise ValueError(remote_path)
    target_path = os.path.join(LOCAL_ROOT, remote_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    _copy_via_part_file(local_path, target_path)


def _get_local(remote_path, local_path):
    source_path = os.path.join(LOCAL_ROOT, remote_path)
    if '..' in remote_path or not os.path.isfile(source_path):
        raise FileNotFoundError(remote_path)
    os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
    _copy_via_part_file(source_path, local_path)


def _ls_local(prefix):
    root = os.path.join(LOCAL_ROOT, prefix)
    if os.path.isfile(root):
        return [{'path': prefix, 'bytes': os.path.getsize(root)}]
    files = []
    for directory, _, names in os.walk(root):
        for name in names:
            if name.endswith('.part'):
                continue
            path = os.path.join(directory, name)
            files.append(
                {'path': os.path.relpath(path, LOCAL_ROOT).replace(os.sep, '/'), 'bytes': os.path.getsize(path)}
            )
    return sorted(files, key=lambda file: file['path'])


def _put_remote(local_path, remote_path):
    with open(local_path, 'rb') as f:
        response = requests.put(f'{URL}/put/{remote_path}', data=f, headers=_headers(), timeout=3600)
    response.raise_for_status()


def _get_remote(remote_path, local_path):
    os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
    with requests.get(f'{URL}/get/{remote_path}', headers=_headers(), stream=True, timeout=3600) as response:
        response.raise_for_status()
        with open(local_path + '.part', 'wb') as f:
            for chunk in response.iter_content(1 << 22):
                f.write(chunk)
    os.replace(local_path + '.part', local_path)


def _is_transient(error):
    if isinstance(error, requests.HTTPError):
        return error.response is not None and error.response.status_code >= 500
    return isinstance(error, (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError))


def _with_retries(transfer, *args):
    """Run transfer(*args), retrying connection errors, timeouts and server errors."""
    for attempt in range(1, TRANSFER_TRIES):
        try:
            transfer(*args)
        except requests.RequestException as error:
            if not _is_transient(error):
                raise
            time.sleep(5 * attempt)
        else:
            return
    transfer(*args)


def put(local_path, remote_path):
    if _is_local():
        _put_local(local_path, remote_path)
    else:
        _with_retries(_put_remote, local_path, remote_path)


def get(remote_path, local_path):
    if _is_local():
        _get_local(remote_path, local_path)
    else:
        _with_retries(_get_remote, remote_path, local_path)


def list_files(prefix=''):
    """Files under prefix as [{'path': ..., 'bytes': ...}]."""
    if _is_local():
        return _ls_local(prefix)
    response = requests.get(f'{URL}/list/{prefix}', headers=_headers(), timeout=300)
    response.raise_for_status()
    return response.json()['files']


def fetch_text(url):
    """Text of a public URL, fetched by the store server for machines that cannot reach the host."""
    if _is_local():  # the store's own compute reaches the internet directly
        response = requests.get(url, headers=USER_AGENT, timeout=300)
    else:
        response = requests.get(f'{URL}/proxy', params={'url': url}, headers=_headers(), timeout=300)
    response.raise_for_status()
    return response.text


def fetch_to_store(url, remote_path):
    """Have the store server download a public URL into the store."""
    if not _is_local():
        response = requests.post(
            f'{URL}/fetch_to_store', params={'url': url, 'dest': remote_path}, headers=_headers(), timeout=3600
        )
        response.raise_for_status()
        return
    target_path = os.path.join(LOCAL_ROOT, remote_path)
    if os.path.isfile(target_path):
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    options = ['-x', '8', '-s', '8', '-c', '--max-tries=5', '--console-log-level=warn']
    destination = ['-d', os.path.dirname(target_path), '-o', os.path.basename(target_path) + '.part']
    result = subprocess.run(['aria2c', *options, *destination, url], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr[-300:])
    os.replace(target_path + '.part', target_path)
