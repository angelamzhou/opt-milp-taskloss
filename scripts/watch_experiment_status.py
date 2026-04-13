import argparse
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from urllib import request


FAILURE_PATTERNS = [
    'TerminatedWorkerError',
    'SIGKILL(-9)',
    'Out Of Memory',
    'Traceback (most recent call last):',
]

PROGRESS_RE = re.compile(
    r'\[\s*(?P<pct>\d+(?:\.\d+)?)%\]\s+completed\s+'
    r'(?P<completed>\d+)/(?P<total>\d+)\s+tasks\s+\|\s+elapsed\s+'
    r'(?P<elapsed>.*?)\s+\|\s+eta\s+(?P<eta>.*?)\s+\|\s+last task\s+(?P<last_task>\S+)'
)


def now_string():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temp_path.replace(path)


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def process_exists(pid):
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def tail_text(path, max_bytes=12000):
    if path is None or not path.exists():
        return ''
    with path.open('rb') as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode('utf-8', errors='replace')


def detect_log_failure(log_text):
    for pattern in FAILURE_PATTERNS:
        if pattern in log_text:
            return pattern
    return None


def parse_log_progress(log_text):
    latest = None
    lines = log_text.splitlines()
    for line in lines:
        match = PROGRESS_RE.search(line)
        if match:
            latest = {
                'percent_complete': float(match.group('pct')),
                'completed_tasks': int(match.group('completed')),
                'total_tasks': int(match.group('total')),
                'last_task_id': match.group('last_task'),
                'elapsed': match.group('elapsed'),
                'eta': match.group('eta'),
            }
    return latest


def detect_log_completion(log_text):
    lines = log_text.splitlines()
    return any(line.startswith('wrote ') for line in lines)


def summarize_snapshot(args, status_payload, pid_alive, log_failure, log_progress=None,
                       log_completed=False, log_path_exists=False):
    if status_payload is None:
        if log_failure:
            state = 'failed'
            message = 'log failure detected'
        elif log_completed:
            state = 'completed'
            message = 'completed from log'
        elif log_progress is not None:
            state = 'running'
            message = 'running | %s/%s tasks | %.2f%% | last=%s' % (
                log_progress['completed_tasks'],
                log_progress['total_tasks'],
                log_progress['percent_complete'],
                log_progress['last_task_id'],
            )
        elif log_path_exists:
            state = 'unknown'
            message = 'log present but no progress line found'
        else:
            state = 'missing'
            message = 'status file missing'
    else:
        state = status_payload.get('status', 'unknown')
        completed = status_payload.get('completed_tasks')
        total = status_payload.get('total_tasks')
        pct = status_payload.get('percent_complete')
        last_task = status_payload.get('last_task_id')
        message = '%s' % state
        if completed is not None and total is not None:
            message += ' | %s/%s tasks' % (completed, total)
        if pct is not None:
            message += ' | %.2f%%' % pct
        if last_task:
            message += ' | last=%s' % last_task

    if log_failure and status_payload is not None:
        state = 'failed'
        message += ' | log_pattern=%s' % log_failure
    elif status_payload and state == 'running' and not pid_alive and args.pid is not None:
        state = 'missing_process'
        message += ' | pid_missing'

    return {'state': state, 'message': message}


def send_slack(webhook_url, text):
    payload = json.dumps({'text': text}).encode('utf-8')
    req = request.Request(
        webhook_url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with request.urlopen(req, timeout=10) as response:
        response.read()


def build_alert_text(args, summary, status_payload, log_failure):
    label = args.job_label or (status_payload or {}).get('output_csv') or str(args.status_path or args.log_path or 'job')
    host = socket.gethostname()
    lines = [
        '[cluster-watch] %s' % label,
        'host: %s' % host,
        'state: %s' % summary['state'],
        'detail: %s' % summary['message'],
    ]
    if status_payload is not None:
        lines.append('status_file: %s' % args.status_path)
        if status_payload.get('log_path'):
            lines.append('log_path: %s' % status_payload.get('log_path'))
        if status_payload.get('error_message'):
            lines.append('error: %s' % status_payload.get('error_message'))
    if log_failure:
        lines.append('log_failure_pattern: %s' % log_failure)
    return '\n'.join(lines)


def should_emit_progress(previous, current, interval):
    if previous is None:
        return True
    prev_pct = previous.get('last_progress_bucket')
    curr_pct = current.get('percent_complete')
    if curr_pct is None:
        return False
    curr_bucket = int(curr_pct // interval) if interval > 0 else int(curr_pct)
    return prev_pct is None or curr_bucket > prev_pct


def poll_snapshot(args):
    status_payload = read_json(args.status_path) if args.status_path else None
    pid = args.pid
    if pid is None and status_payload is not None:
        pid = status_payload.get('pid')
    pid_alive = process_exists(pid) if pid is not None else True

    log_path = Path(args.log_path) if args.log_path else None
    if log_path is None and status_payload and status_payload.get('log_path'):
        log_path = Path(status_payload['log_path'])
    log_text = tail_text(log_path, max_bytes=args.log_tail_bytes)
    log_path_exists = bool(log_path and log_path.exists())
    log_failure = detect_log_failure(log_text)
    log_progress = parse_log_progress(log_text)
    log_completed = detect_log_completion(log_text)
    summary = summarize_snapshot(
        args,
        status_payload,
        pid_alive,
        log_failure,
        log_progress=log_progress,
        log_completed=log_completed,
        log_path_exists=log_path_exists,
    )
    if status_payload is None and log_progress is not None:
        percent_complete = log_progress.get('percent_complete')
    else:
        percent_complete = None if status_payload is None else status_payload.get('percent_complete')
    return {
        'summary': summary,
        'status_payload': status_payload,
        'pid_alive': pid_alive,
        'log_failure': log_failure,
        'percent_complete': percent_complete,
        'log_progress': log_progress,
        'log_completed': log_completed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status-path', default=None)
    parser.add_argument('--log-path', default=None)
    parser.add_argument('--pid', type=int, default=None)
    parser.add_argument('--job-label', default=None)
    parser.add_argument('--poll-seconds', type=int, default=120)
    parser.add_argument('--progress-interval-pct', type=float, default=10.0)
    parser.add_argument('--state-file', default=None)
    parser.add_argument('--slack-webhook-url', default=None)
    parser.add_argument('--slack-webhook-env', default='SLACK_WEBHOOK_URL')
    parser.add_argument('--log-tail-bytes', type=int, default=12000)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    if args.status_path is None and args.log_path is None and args.pid is None:
        raise SystemExit('provide at least one of --status-path, --log-path, or --pid')

    if args.status_path is not None:
        args.status_path = Path(args.status_path)
    if args.log_path is not None:
        args.log_path = Path(args.log_path)
    state_file = Path(args.state_file) if args.state_file else None
    if state_file is None:
        base = args.status_path or args.log_path or Path('watcher')
        state_file = Path(str(base) + '.watcher.json')

    webhook_url = args.slack_webhook_url or os.environ.get(args.slack_webhook_env)
    previous_state = read_json(state_file) or {}

    while True:
        snapshot = poll_snapshot(args)
        summary = snapshot['summary']
        current_state = {
            'state': summary['state'],
            'message': summary['message'],
            'last_progress_bucket': (
                int(snapshot['percent_complete'] // args.progress_interval_pct)
                if snapshot['percent_complete'] is not None and args.progress_interval_pct > 0
                else None
            ),
            'timestamp': now_string(),
        }

        emit = False
        if previous_state.get('state') != current_state['state']:
            emit = True
        elif summary['state'] == 'running' and should_emit_progress(previous_state, snapshot, args.progress_interval_pct):
            emit = True

        if emit:
            text = build_alert_text(args, summary, snapshot['status_payload'], snapshot['log_failure'])
            print(text, flush=True)
            if webhook_url:
                try:
                    send_slack(webhook_url, text)
                except Exception as exc:
                    print('slack alert failed: %s' % exc, file=sys.stderr, flush=True)

        atomic_write_json(state_file, current_state)
        previous_state = current_state

        if args.once or summary['state'] in ['completed', 'failed', 'interrupted', 'missing_process']:
            break
        time.sleep(max(5, args.poll_seconds))


if __name__ == '__main__':
    main()
