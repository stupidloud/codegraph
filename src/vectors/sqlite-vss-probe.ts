import * as childProcess from 'child_process';
import * as path from 'path';

export interface SqliteVssProbeResult {
  available: boolean;
  checkedAt: number;
  loadablePaths?: SqliteVssLoadablePaths;
  reason?: string;
}

export interface SqliteVssLoadablePaths {
  vector: string;
  vss: string;
}

type SpawnSyncFn = typeof childProcess.spawnSync;

/**
 * Probe sqlite-vss in a child process because native extension failures such as
 * SIGILL kill the process and cannot be caught with try/catch.
 */
export function probeSqliteVss(
  timeoutMs: number = 5000,
  spawnSyncFn: SpawnSyncFn = childProcess.spawnSync
): SqliteVssProbeResult {
  const packageRoot = path.resolve(__dirname, '..', '..');
  const childScript = path.join(__dirname, 'sqlite-vss-probe-child.js');
  const result = spawnSyncFn(process.execPath, [childScript, packageRoot], {
    encoding: 'utf8',
    timeout: timeoutMs,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  const checkedAt = Date.now();
  if (result.status === 0) {
    try {
      const loadablePaths = JSON.parse(String(result.stdout || '')) as SqliteVssLoadablePaths;
      if (typeof loadablePaths.vector === 'string' && typeof loadablePaths.vss === 'string') {
        return { available: true, checkedAt, loadablePaths };
      }
    } catch {
      // Treat malformed probe output as unavailable; runtime should not guess.
    }

    return {
      available: false,
      checkedAt,
      reason: 'sqlite-vss probe succeeded but did not return loadable paths',
    };
  }

  const reason = result.error
    ? result.error.message
    : result.signal
      ? `probe exited via signal ${result.signal}`
      : (result.stderr || result.stdout || `probe exited with status ${result.status}`).trim();

  return {
    available: false,
    checkedAt,
    reason: reason.slice(0, 1000),
  };
}
