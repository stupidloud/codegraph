import * as childProcess from 'child_process';

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
  const script = `
    const fs = require('fs');
    const os = require('os');
    const path = require('path');

    function stripPlatformExtensionSuffix(loadablePath) {
      const ext = path.extname(loadablePath);
      return ext === '.so' || ext === '.dylib' || ext === '.dll'
        ? loadablePath.slice(0, -ext.length)
        : loadablePath;
    }

    function loadWithFallback(db, loadablePath) {
      const withoutSuffix = stripPlatformExtensionSuffix(loadablePath);
      const candidates = withoutSuffix === loadablePath ? [loadablePath] : [withoutSuffix, loadablePath];
      let lastError;
      for (const candidate of candidates) {
        try {
          db.loadExtension(candidate);
          return candidate;
        } catch (error) {
          lastError = error;
        }
      }
      throw lastError;
    }

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'codegraph-vss-probe-'));
    const dbPath = path.join(tmpDir, 'probe.db');
    let db;
    let exitCode = 0;
    try {
      const Database = require('better-sqlite3');
      const vss = require('sqlite-vss');
      db = new Database(dbPath);
      const getVectorLoadablePath = vss.getVectorLoadablePath || vss.default?.getVectorLoadablePath;
      const getVssLoadablePath = vss.getVssLoadablePath || vss.default?.getVssLoadablePath;
      if (typeof db.loadExtension !== 'function') {
        throw new Error('SQLite connection does not support loadExtension');
      }
      if (typeof getVectorLoadablePath !== 'function' || typeof getVssLoadablePath !== 'function') {
        throw new Error('sqlite-vss loadable path functions not found');
      }
      const vectorPath = loadWithFallback(db, getVectorLoadablePath());
      const vssPath = loadWithFallback(db, getVssLoadablePath());
      db.exec('CREATE VIRTUAL TABLE probe_vectors USING vss0(embedding(3));');
      process.stdout.write(JSON.stringify({ vector: vectorPath, vss: vssPath }));
    } catch (error) {
      process.stderr.write(error && error.stack ? error.stack : String(error));
      exitCode = 1;
    } finally {
      try { if (db) db.close(); } catch {}
      try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
    }
    process.exit(exitCode);
  `;

  const result = spawnSyncFn(process.execPath, ['-e', script], {
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
