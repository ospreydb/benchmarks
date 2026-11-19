use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio_postgres::{Client, NoTls, Statement};

use crate::HttpLoadReport;

pub(crate) struct PgLoadConfig<'a> {
    pub(crate) dsn: &'a str,
    pub(crate) table: &'a str,
    pub(crate) col_type: &'a str,
    pub(crate) ef_search: usize,
    pub(crate) limit: usize,
    pub(crate) payloads_file: &'a Path,
    pub(crate) warmup_payloads_file: Option<&'a Path>,
    pub(crate) workers: usize,
    pub(crate) out_json: &'a Path,
}

/// Validates that a PostgreSQL table identifier matches `^[A-Za-z_][A-Za-z0-9_]*$`.
pub(crate) fn validate_table_name(table: &str) -> Result<(), String> {
    let mut chars = table.chars();
    let Some(first) = chars.next() else {
        return Err("table name must not be empty".to_string());
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(format!(
            "invalid table name '{table}': must start with ASCII letter or underscore"
        ));
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!(
            "invalid table name '{table}': must contain only ASCII alphanumeric or underscore characters"
        ));
    }
    Ok(())
}

/// Validates that the pgvector column type is strictly `"vector"` or `"halfvec"`.
pub(crate) fn validate_col_type(col_type: &str) -> Result<(), String> {
    match col_type {
        "vector" | "halfvec" => Ok(()),
        other => Err(format!(
            "unsupported --col-type '{other}' (expected vector or halfvec)"
        )),
    }
}

/// Validates identifiers and constructs the parameterized pgvector ANN query.
pub(crate) fn build_search_sql(
    table: &str,
    col_type: &str,
    limit: usize,
) -> Result<String, String> {
    validate_table_name(table)?;
    validate_col_type(col_type)?;
    Ok(format!(
        "SELECT id FROM \"{table}\" ORDER BY emb <=> $1::text::{col_type} LIMIT {limit}"
    ))
}

/// Loads non-empty, trimmed lines from a file (e.g. pgvector text literals).
pub(crate) fn load_nonempty_lines(path: &Path) -> Result<Vec<String>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read {}: {e}", path.display()))?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
    }
    Ok(lines)
}

async fn connect_and_prepare_worker(
    dsn: &str,
    set_sql: &str,
    query_sql: &str,
    warmup_lit: Option<&str>,
) -> Result<(Client, Statement), String> {
    let (client, connection) = tokio_postgres::connect(dsn, NoTls)
        .await
        .map_err(|e| format!("postgres connect failed: {e}"))?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("postgres connection error: {e}");
        }
    });

    client
        .batch_execute(set_sql)
        .await
        .map_err(|e| format!("execute '{set_sql}' failed: {e}"))?;
    let stmt = client
        .prepare(query_sql)
        .await
        .map_err(|e| format!("prepare query failed: {e}"))?;

    if let Some(lit) = warmup_lit {
        let _ = client
            .query(&stmt, &[&lit])
            .await
            .map_err(|e| format!("warmup query failed: {e}"))?;
    }

    Ok((client, stmt))
}

/// Runs the async `tokio-postgres` load test and writes the JSON report.
pub(crate) fn run_pg_load(cfg: PgLoadConfig<'_>) -> Result<(), String> {
    let query_sql = build_search_sql(cfg.table, cfg.col_type, cfg.limit)?;
    let set_sql = format!("SET hnsw.ef_search = {};", cfg.ef_search);
    let payloads = load_nonempty_lines(cfg.payloads_file)?;
    let warmup_payloads = match cfg.warmup_payloads_file {
        Some(path) => load_nonempty_lines(path)?,
        None => Vec::new(),
    };

    let total = payloads.len();
    let payloads = Arc::new(payloads);
    let n_workers = cfg.workers.max(1).min(total.max(1));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;

    let dsn = cfg.dsn.to_string();
    let report = rt.block_on(async move {
        let mut setup_handles = Vec::with_capacity(n_workers);
        for wi in 0..n_workers {
            let dsn_w = dsn.clone();
            let set_w = set_sql.clone();
            let query_w = query_sql.clone();
            let warm_w = if warmup_payloads.is_empty() {
                None
            } else {
                Some(warmup_payloads[wi % warmup_payloads.len()].clone())
            };
            setup_handles.push(tokio::spawn(async move {
                connect_and_prepare_worker(&dsn_w, &set_w, &query_w, warm_w.as_deref()).await
            }));
        }

        let mut worker_conns = Vec::with_capacity(n_workers);
        for h in setup_handles {
            worker_conns.push(h.await.map_err(|e| e.to_string())??);
        }

        type PgQueryResult = (Vec<u64>, f64);
        type PgResultsBuffer = Arc<tokio::sync::Mutex<Vec<Option<PgQueryResult>>>>;

        let results: PgResultsBuffer = Arc::new(tokio::sync::Mutex::new(vec![None; total]));
        let next_idx = Arc::new(AtomicUsize::new(0));

        let wall_start = Instant::now();
        let mut handles = Vec::with_capacity(n_workers);
        for (client, stmt) in worker_conns {
            let pl = Arc::clone(&payloads);
            let idx_gen = Arc::clone(&next_idx);
            let res_slot = Arc::clone(&results);
            handles.push(tokio::spawn(async move {
                let mut local: Vec<(usize, Vec<u64>, f64)> = Vec::new();
                loop {
                    let idx = idx_gen.fetch_add(1, Ordering::Relaxed);
                    if idx >= pl.len() {
                        break;
                    }
                    let lit = &pl[idx];
                    let t0 = Instant::now();
                    let rows = client
                        .query(&stmt, &[lit])
                        .await
                        .map_err(|e| format!("postgres query failed: {e}"))?;
                    let client_ms = t0.elapsed().as_secs_f64() * 1000.0;

                    let mut ids = Vec::with_capacity(rows.len());
                    for row in rows {
                        let raw_id: i64 = row
                            .try_get(0)
                            .map_err(|e| format!("read id column failed: {e}"))?;
                        let id = u64::try_from(raw_id)
                            .map_err(|_| format!("negative id {raw_id} returned from postgres"))?;
                        ids.push(id);
                    }
                    local.push((idx, ids, client_ms));
                }

                let mut guard = res_slot.lock().await;
                for (idx, ids, cms) in local {
                    guard[idx] = Some((ids, cms));
                }
                Ok::<(), String>(())
            }));
        }

        for h in handles {
            h.await.map_err(|e| e.to_string())??;
        }
        let wall_s = wall_start.elapsed().as_secs_f64();

        let guard = results.lock().await;
        let mut doc_ids = Vec::with_capacity(total);
        let mut client_ms = Vec::with_capacity(total);
        for item in guard.iter() {
            let (ids, cms) = item
                .as_ref()
                .ok_or_else(|| "missing pg-load result slot".to_string())?;
            doc_ids.push(ids.clone());
            client_ms.push(*cms);
        }

        Ok::<HttpLoadReport, String>(HttpLoadReport {
            doc_ids,
            client_ms,
            server_ms: Vec::new(),
            wall_s,
        })
    })?;

    if let Some(parent) = cfg.out_json.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let body = serde_json::to_string(&report).map_err(|e| e.to_string())?;
    std::fs::write(cfg.out_json, body).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_table_names() {
        assert!(validate_table_name("items").is_ok());
        assert!(validate_table_name("_pg_vec_01").is_ok());
        assert!(validate_table_name("Sift1M_halfvec").is_ok());

        assert!(validate_table_name("").is_err());
        assert!(validate_table_name("123items").is_err());
        assert!(validate_table_name("items-v1").is_err());
        assert!(validate_table_name("items; DROP TABLE x").is_err());
        assert!(validate_table_name("public.items").is_err());
        assert!(validate_table_name("téb").is_err());
    }

    #[test]
    fn validates_col_types() {
        assert!(validate_col_type("vector").is_ok());
        assert!(validate_col_type("halfvec").is_ok());

        assert!(validate_col_type("").is_err());
        assert!(validate_col_type("bit").is_err());
        assert!(validate_col_type("VECTOR").is_err());
        assert!(validate_col_type("vector(128)").is_err());
    }

    #[test]
    fn builds_parameterized_search_sql() {
        let sql_vec = build_search_sql("bench_docs", "vector", 10).expect("valid vector sql");
        assert_eq!(
            sql_vec,
            "SELECT id FROM \"bench_docs\" ORDER BY emb <=> $1::text::vector LIMIT 10"
        );

        let sql_half = build_search_sql("bench_docs_fp16", "halfvec", 25).expect("valid halfvec");
        assert_eq!(
            sql_half,
            "SELECT id FROM \"bench_docs_fp16\" ORDER BY emb <=> $1::text::halfvec LIMIT 25"
        );

        assert!(build_search_sql("bad-name", "vector", 10).is_err());
        assert!(build_search_sql("bench_docs", "bad_type", 10).is_err());
    }

    #[test]
    fn loads_nonempty_lines_trimming_whitespace() {
        let dir =
            std::env::temp_dir().join(format!("osprey_pg_load_lines_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("payloads.txt");
        let content = "\n  [0.1, 0.2]  \n\n   \t  \n[0.3,0.4]\n";
        std::fs::write(&path, content).expect("write payloads file");

        let loaded = load_nonempty_lines(&path).expect("load nonempty lines");
        assert_eq!(
            loaded,
            vec!["[0.1, 0.2]".to_string(), "[0.3,0.4]".to_string()]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
