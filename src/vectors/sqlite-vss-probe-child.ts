import * as fs from 'fs';
import { createRequire } from 'module';
import * as os from 'os';
import * as path from 'path';

function stripPlatformExtensionSuffix(loadablePath: string): string {
  const ext = path.extname(loadablePath);
  return ext === '.so' || ext === '.dylib' || ext === '.dll'
    ? loadablePath.slice(0, -ext.length)
    : loadablePath;
}

function loadWithFallback(
  db: ProbeDatabase,
  loadablePath: string
): string {
  const withoutSuffix = stripPlatformExtensionSuffix(loadablePath);
  const candidates = withoutSuffix === loadablePath ? [loadablePath] : [withoutSuffix, loadablePath];
  let lastError: unknown;
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

interface ProbeDatabase {
  loadExtension(path: string): void;
  exec(sql: string): void;
  close(): void;
}

function main(): number {
  const packageRoot = process.argv[2];
  if (!packageRoot) {
    process.stderr.write('Missing package root for sqlite-vss probe child');
    return 1;
  }

  const packageRequire = createRequire(path.join(packageRoot, 'package.json'));
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'codegraph-vss-probe-'));
  const dbPath = path.join(tmpDir, 'probe.db');
  let db: ProbeDatabase | undefined;

  try {
    const Database = packageRequire('better-sqlite3') as new (filename: string) => ProbeDatabase;
    const vss = packageRequire('sqlite-vss') as {
      getVectorLoadablePath?: () => string;
      getVssLoadablePath?: () => string;
      default?: {
        getVectorLoadablePath?: () => string;
        getVssLoadablePath?: () => string;
      };
    };

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
    return 0;
  } catch (error) {
    process.stderr.write(error instanceof Error ? error.stack || error.message : String(error));
    return 1;
  } finally {
    try {
      db?.close();
    } catch {}
    try {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    } catch {}
  }
}

process.exit(main());
