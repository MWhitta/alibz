#!/usr/bin/env python3
"""Read deployed optimization readiness without moving or firing hardware."""
import argparse
import subprocess

REMOTE = r'''
import http.client, json, os, socket, time
class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(os.path.expanduser('~/.local/state/pantheum/alibz-gateway/alibz.sock'))
def get(path):
    c = UnixHTTP('localhost', timeout=15)
    c.request('GET', path, headers={'X-Portal-User': 'mwhittaker@lbl.gov'})
    response = c.getresponse()
    value = json.loads(response.read())
    c.close()
    if response.status != 200:
        raise RuntimeError((path, response.status, value))
    return value
motion = get('/api/motion/status?touch=1')
deadline = time.monotonic() + 10
while not motion.get('observed_at') and time.monotonic() < deadline:
    time.sleep(1)
    motion = get('/api/motion/status?touch=1')
readiness = get('/api/acquire/status')
print(json.dumps({
    'checked_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'hardware_checkout': get('/api/reservation?resource=hardware'),
    'motion': motion,
    'readiness': {k: v for k, v in readiness.items() if k != 'presets'},
    'optimization': get('/api/optimization'),
}, indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    compile(REMOTE, '<remote readiness check>', 'exec')
    if args.dry_run:
        print('SSH moissanite: GET motion/status?touch=1, acquire/status, '
              'reservation?resource=hardware, optimization. No POST requests.')
        return
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    'moissanite', 'python3 -'], input=REMOTE, text=True,
                   check=True, timeout=90)


if __name__ == '__main__':
    main()
