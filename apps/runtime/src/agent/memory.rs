use chrono::Utc;
use rusqlite::{params, Connection};
use std::path::PathBuf;

#[derive(Clone)]
pub struct MemoryStore {
    path: PathBuf,
}

impl MemoryStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn init(&self) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let conn = self.conn()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                scope TEXT NOT NULL,
                run_id TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(content, memory_id UNINDEXED);
            CREATE TABLE IF NOT EXISTS summaries (
                run_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .map_err(|error| error.to_string())
    }

    pub fn write(&self, scope: &str, run_id: Option<&str>, content: &str) -> Result<(), String> {
        self.init()?;
        let now = Utc::now().to_rfc3339();
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO memories (scope, run_id, content, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)",
            params![scope, run_id, content, now],
        )
        .map_err(|error| error.to_string())?;
        let id = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO memory_fts (content, memory_id) VALUES (?1, ?2)",
            params![content, id],
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<String>, String> {
        self.init()?;
        let Some(expr) = fts_query(query) else {
            return Ok(Vec::new());
        };
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT m.content
                 FROM memory_fts f
                 JOIN memories m ON m.id = f.memory_id
                 WHERE memory_fts MATCH ?1
                 ORDER BY m.updated_at DESC
                 LIMIT ?2",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map(params![expr, limit as i64], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())
    }

    pub fn summary(&self, run_id: &str) -> Result<Option<String>, String> {
        self.init()?;
        self.conn()?
            .query_row(
                "SELECT content FROM summaries WHERE run_id = ?1",
                params![run_id],
                |row| row.get(0),
            )
            .map(Some)
            .or_else(|error| match error {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.to_string()),
            })
    }

    fn conn(&self) -> Result<Connection, String> {
        Connection::open(&self.path).map_err(|error| error.to_string())
    }
}

fn fts_query(query: &str) -> Option<String> {
    let terms = query
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|term| term.len() >= 3)
        .take(8)
        .map(|term| format!("\"{term}\""))
        .collect::<Vec<_>>();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}
