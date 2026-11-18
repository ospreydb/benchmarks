use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use clap::{Parser, Subcommand};
use memmap2::Mmap;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use osprey_core::{
    cast_minipage_directory, AlignedBuffer, MinipageDirectoryEntry, OsiIndex, OsiMappedView,
    OspreyError, OsrStore, OsvHeader, QuantizedQueryLut, SearchOptions,
};
use osprey_simd::{SimdBackend, SimdEngine};
use osprey_storage::{
    build_object_store, MinipageLruCache, RefineStage, RemoteStoreConfig, WarpConfig,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(name = "osprey-bench-runner")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Evaluates S3Vec (.osi + .osv over local or S3/MinIO) across (nprobe, refine_m) Pareto grid.
    SearchEval {
        #[arg(long)]
        osi: PathBuf,
        #[arg(long)]
        slabs: String,
        #[arg(long)]
        queries_npy: PathBuf,
        #[arg(long)]
        gt_npy: PathBuf,
        #[arg(long, default_value = "16")]
        nprobe_list: String,
        /// Comma-separated Stage-1 candidate pools. `0` selects the policy default
        /// (2000 for `orig_dim <= 256`, 300 otherwise).
        #[arg(long, default_value = "200")]
        refine_list: String,
        /// `0` = adaptive (2x + 128 f16 floor when dim <= 256), `1` = disabled, `N > 1` = forced Nx.
        #[arg(long, default_value_t = 0)]
        refine_multiplier: usize,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value_t = 100)]
        max_queries: usize,
        #[arg(long, default_value_t = 0)]
        cache_mb: usize,
        #[arg(long)]
        no_warp: bool,
        #[arg(long)]
        out_json: PathBuf,
    },
    /// Serves the S3Vec index over local HTTP for controlled client-server benchmarks.
    Serve {
        #[arg(long)]
        osi: PathBuf,
        #[arg(long)]
        slabs: String,
        #[arg(long, default_value = "127.0.0.1:18080")]
        bind: String,
        #[arg(long, default_value_t = 16)]
        cache_mb: usize,
        #[arg(long)]
        no_warp: bool,
    },
    /// Evaluates CPU LUT-Rank (.osr) against MS MARCO / BEIR queries and qrels.
    LutrankEval {
        #[arg(long)]
        osr: PathBuf,
        #[arg(long)]
        queries_json: PathBuf,
        #[arg(long, default_value_t = 200)]
        candidates_per_query: usize,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value_t = 100)]
        max_queries: usize,
        #[arg(long)]
        out_json: PathBuf,
    },
}

#[derive(Debug, Serialize)]
struct SearchEvalPoint {
    nprobe: usize,
    refine_m: usize,
    requested_nprobe: usize,
    effective_nprobe: usize,
    requested_candidates: usize,
    effective_candidates: usize,
    refine_limit: usize,
    stage1_limit: usize,
    refine_multiplier: usize,
    low_dim_oversample: bool,
    auto_candidates: bool,
    orig_dim: u16,
    warp_enabled: bool,
    queries_evaluated: usize,
    recall_at_10: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    qps: f64,
    stage1_mean_ms: f64,
    warp_plan_mean_us: f64,
    stage2_3_mean_ms: f64,
    avg_raw_pages: f64,
    avg_s3_gets: f64,
    warp_coalesce_factor: f64,
    avg_bytes_fetched: f64,
    osi_bytes: u64,
    osv_bytes: u64,
    vm_hwm_mb: f64,
    simd_tier: String,
}

#[derive(Debug, Deserialize)]
struct ServerSearchRequest {
    query: Vec<f32>,
    #[serde(default = "default_server_nprobe")]
    nprobe: usize,
    #[serde(default)]
    refine_m: usize,
    #[serde(default = "default_server_top_k")]
    top_k: usize,
    #[serde(default)]
    refine_multiplier: usize,
}

fn default_server_nprobe() -> usize {
    32
}

fn default_server_top_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
struct ServerSearchResponse {
    doc_ids: Vec<u64>,
    server_ms: f64,
    stage1_ms: f64,
    refine_ms: f64,
    candidates: usize,
    effective_nprobe: usize,
    effective_candidates: usize,
    stage1_limit: usize,
    low_dim_oversample: bool,
}

struct ServerState {
    osi: Arc<OsiIndex>,
    orig_dim: u16,
    refine_stage: Arc<RefineStage>,
    directory: Vec<MinipageDirectoryEntry>,
    cache: Arc<MinipageLruCache>,
    store: Arc<dyn ObjectStore>,
    obj_path: ObjectPath,
    remote_cfg: RemoteStoreConfig,
}

async fn handle_server_search(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ServerSearchRequest>,
) -> Result<Json<ServerSearchResponse>, (StatusCode, String)> {
    if req.query.len() != usize::from(state.orig_dim) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "expected query dimension {}, got {}",
                state.orig_dim,
                req.query.len()
            ),
        ));
    }
    if req.top_k == 0 {
        return Err((StatusCode::BAD_REQUEST, "top_k must be > 0".to_string()));
    }

    let plan = SearchOptions {
        nprobe: req.nprobe,
        candidates: req.refine_m,
        top_k: req.top_k,
        rerank_top_m: 0,
        refine_multiplier: req.refine_multiplier,
    }
    .resolve(state.orig_dim);

    let t0 = Instant::now();
    let osi = Arc::clone(&state.osi);
    let query = req.query.clone();
    let t_s1 = Instant::now();
    let hits = tokio::task::spawn_blocking(move || {
        let simd = SimdEngine::new_auto();
        osi.search_candidates(&query, plan.effective_nprobe, plan.stage1_limit, &simd)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let stage1_ms = t_s1.elapsed().as_secs_f64() * 1000.0;

    let t_refine = Instant::now();
    let (refined, _stats) = state
        .refine_stage
        .refine_scored_async(
            &req.query,
            &hits,
            req.top_k,
            &state.directory,
            &state.cache,
            &state.store,
            &state.obj_path,
            &state.remote_cfg,
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let refine_ms = t_refine.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(ServerSearchResponse {
        doc_ids: refined.iter().map(|r| r.doc_id).collect(),
        server_ms: t0.elapsed().as_secs_f64() * 1000.0,
        stage1_ms,
        refine_ms,
        candidates: hits.len(),
        effective_nprobe: plan.effective_nprobe,
        effective_candidates: plan.effective_candidates,
        stage1_limit: plan.stage1_limit,
        low_dim_oversample: plan.low_dim_oversample,
    }))
}

#[derive(Debug, Deserialize)]
struct BeirQuery {
    query_id: String,
    tokens: Vec<Vec<f32>>,
    #[serde(default)]
    qrels: HashMap<String, i64>,
}

#[derive(Debug, Serialize)]
struct LutrankEvalReport {
    osr_path: String,
    osr_bytes: u64,
    simd_tier: String,
    queries_evaluated: usize,
    candidates_per_query: usize,
    lut_synth_mean_us: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    qps: f64,
    ndcg_at_10: f64,
    mrr_at_10: f64,
    vm_hwm_mb: f64,
}

fn read_vm_hwm_mb() -> f64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(kb_str) = rest.split_whitespace().next() {
                if let Ok(kb) = kb_str.parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }
    0.0
}

fn read_npy_2d_raw(path: &Path) -> Result<(Vec<usize>, String, Vec<u8>), String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a valid .npy file", path.display()));
    }
    let major = magic[6];
    let header_len = if major == 1 {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf).map_err(|e| e.to_string())?;
        usize::from(u16::from_le_bytes(buf))
    } else {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).map_err(|e| e.to_string())?;
        usize::try_from(u32::from_le_bytes(buf)).map_err(|e| e.to_string())?
    };

    let mut hdr_bytes = vec![0u8; header_len];
    file.read_exact(&mut hdr_bytes).map_err(|e| e.to_string())?;
    let hdr = String::from_utf8_lossy(&hdr_bytes);

    let descr = extract_dict_str(&hdr, "'descr'")?;
    let shape = extract_shape(&hdr)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).map_err(|e| e.to_string())?;
    Ok((shape, descr, data))
}

fn extract_dict_str(hdr: &str, key: &str) -> Result<String, String> {
    let pos = hdr
        .find(key)
        .ok_or_else(|| format!("missing {key} in {hdr}"))?;
    let after = &hdr[pos + key.len()..];
    let q1 = after
        .find('\'')
        .ok_or_else(|| "missing quote".to_string())?;
    let rest = &after[q1 + 1..];
    let q2 = rest
        .find('\'')
        .ok_or_else(|| "missing end quote".to_string())?;
    Ok(rest[..q2].to_string())
}

fn extract_shape(hdr: &str) -> Result<Vec<usize>, String> {
    let pos = hdr
        .find("'shape'")
        .ok_or_else(|| "missing shape".to_string())?;
    let after = &hdr[pos..];
    let p1 = after.find('(').ok_or_else(|| "missing (".to_string())?;
    let p2 = after.find(')').ok_or_else(|| "missing )".to_string())?;
    let inner = &after[p1 + 1..p2];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            dims.push(trimmed.parse::<usize>().map_err(|e| e.to_string())?);
        }
    }
    Ok(dims)
}

fn load_f32_matrix(path: &Path) -> Result<(usize, usize, Vec<f32>), String> {
    let (shape, descr, data) = read_npy_2d_raw(path)?;
    if shape.len() != 2 {
        return Err(format!(
            "expected 2D matrix in {}, got {:?}",
            path.display(),
            shape
        ));
    }
    if descr != "<f4" && descr != "f4" {
        return Err(format!("expected <f4 in {}, got {descr}", path.display()));
    }
    let rows = shape[0];
    let cols = shape[1];
    if data.len() < rows * cols * 4 {
        return Err("truncated f32 npy data".to_string());
    }
    let mut out = Vec::with_capacity(rows * cols);
    for chunk in data[..rows * cols * 4].as_chunks::<4>().0 {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((rows, cols, out))
}

fn load_i64_matrix(path: &Path) -> Result<(usize, usize, Vec<i64>), String> {
    let (shape, descr, data) = read_npy_2d_raw(path)?;
    if shape.len() != 2 {
        return Err(format!(
            "expected 2D matrix in {}, got {:?}",
            path.display(),
            shape
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    let mut out = Vec::with_capacity(rows * cols);
    if descr == "<i8" || descr == "i8" {
        for chunk in data[..rows * cols * 8].as_chunks::<8>().0 {
            out.push(i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
    } else if descr == "<i4" || descr == "i4" {
        for chunk in data[..rows * cols * 4].as_chunks::<4>().0 {
            out.push(i64::from(i32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ])));
        }
    } else {
        return Err(format!(
            "unsupported int dtype {descr} in {}",
            path.display()
        ));
    }
    Ok((rows, cols, out))
}

fn percentile(sorted: &[f64], p: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) * p) / 100;
    sorted[idx]
}

async fn load_osv_metadata(
    store: &Arc<dyn ObjectStore>,
    obj_path: &ObjectPath,
) -> Result<(Vec<MinipageDirectoryEntry>, Option<Vec<u8>>, u64), String> {
    let meta = store
        .head(obj_path)
        .await
        .map_err(|e| format!("head: {e}"))?;
    let osv_size = u64::try_from(meta.size).unwrap_or(0);

    let hdr_bytes = store
        .get_range(obj_path, 0..64)
        .await
        .map_err(|e| format!("get header: {e}"))?;
    let aligned_hdr = AlignedBuffer::from_slice(&hdr_bytes).map_err(|e| e.to_string())?;
    let hdr = *OsvHeader::from_bytes(aligned_hdr.as_slice()).map_err(|e| e.to_string())?;

    let mut ranges: Vec<Range<usize>> = Vec::with_capacity(2);
    let has_dict = hdr.zstd_dict_len > 0;
    if has_dict {
        let start = usize::try_from(hdr.zstd_dict_offset).map_err(|e| e.to_string())?;
        let len = usize::try_from(hdr.zstd_dict_len).map_err(|e| e.to_string())?;
        ranges.push(start..start + len);
    }
    let dir_start = usize::try_from(hdr.directory_offset).map_err(|e| e.to_string())?;
    let dir_len = usize::try_from(hdr.num_minipages)
        .map_err(|e| e.to_string())?
        .checked_mul(std::mem::size_of::<MinipageDirectoryEntry>())
        .ok_or_else(|| "dir overflow".to_string())?;
    if dir_len > 0 {
        ranges.push(dir_start..dir_start + dir_len);
    }

    let fetched = store
        .get_ranges(obj_path, &ranges)
        .await
        .map_err(|e| format!("get metadata: {e}"))?;
    let mut idx = 0;
    let zstd_dict = if has_dict {
        let v = fetched[idx].to_vec();
        idx += 1;
        Some(v)
    } else {
        None
    };
    let directory = if dir_len > 0 {
        let aligned_dir = AlignedBuffer::from_slice(&fetched[idx]).map_err(|e| e.to_string())?;
        cast_minipage_directory(aligned_dir.as_slice(), hdr.num_minipages)
            .map_err(|e| e.to_string())?
            .to_vec()
    } else {
        Vec::new()
    };
    Ok((directory, zstd_dict, osv_size))
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|p| p.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect()
}

fn normalize_slabs_uri(slabs: &str) -> Result<String, String> {
    if slabs.contains("://") {
        Ok(slabs.to_string())
    } else {
        let abs = std::fs::canonicalize(slabs).map_err(|e| format!("canonicalize {slabs}: {e}"))?;
        Ok(format!("file://{}", abs.display()))
    }
}

fn run_server(
    osi_path: &Path,
    slabs: &str,
    bind: &str,
    cache_mb: usize,
    no_warp: bool,
) -> Result<(), String> {
    let file = File::open(osi_path).map_err(|e| format!("open {}: {e}", osi_path.display()))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
    let _view = OsiMappedView::from_bytes(&mmap).map_err(|e| e.to_string())?;
    let osi = OsiIndex::from_bytes(&mmap).map_err(|e| e.to_string())?;
    let orig_dim = osi.header().orig_dim;

    let slabs_uri = normalize_slabs_uri(slabs)?;
    let remote_cfg = RemoteStoreConfig::default();
    let (store, obj_path) =
        build_object_store(&slabs_uri, &remote_cfg).map_err(|e| e.to_string())?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;

    let (directory, zstd_dict, _osv_bytes) = rt.block_on(load_osv_metadata(&store, &obj_path))?;

    let warp_cfg = if no_warp {
        WarpConfig {
            tau_req: 0.0,
            tau_byte: 1.0,
            max_coalesce_bytes: 0,
        }
    } else {
        WarpConfig::default()
    };

    let state = Arc::new(ServerState {
        osi: Arc::new(osi),
        orig_dim,
        refine_stage: Arc::new(RefineStage::new(0, warp_cfg, zstd_dict)),
        directory,
        cache: Arc::new(MinipageLruCache::new(cache_mb.saturating_mul(1024 * 1024))),
        store,
        obj_path,
        remote_cfg,
    });

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/search", post(handle_server_search))
        .with_state(state);

    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(bind)
            .await
            .map_err(|e| format!("bind {bind}: {e}"))?;
        println!("osprey-bench-runner HTTP server listening on http://{bind} (dim={orig_dim})");
        axum::serve(listener, app)
            .await
            .map_err(|e| format!("serve: {e}"))
    })
}

#[allow(clippy::too_many_arguments)]
fn run_search_eval(
    osi_path: &Path,
    slabs: &str,
    queries_npy: &Path,
    gt_npy: &Path,
    nprobe_list: &[usize],
    refine_list: &[usize],
    refine_multiplier: usize,
    top_k: usize,
    max_queries: usize,
    cache_mb: usize,
    no_warp: bool,
    out_json: &Path,
) -> Result<(), String> {
    let osi_bytes = std::fs::metadata(osi_path)
        .map_err(|e| e.to_string())?
        .len();
    let file = File::open(osi_path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
    let _view = OsiMappedView::from_bytes(&mmap).map_err(|e| e.to_string())?;
    let osi = OsiIndex::from_bytes(&mmap).map_err(|e| e.to_string())?;
    let orig_dim = osi.header().orig_dim;

    let (num_q, q_dim, q_flat) = load_f32_matrix(queries_npy)?;
    let (gt_rows, gt_cols, gt_flat) = load_i64_matrix(gt_npy)?;
    let q_count = num_q.min(gt_rows).min(max_queries);

    let slabs_uri = normalize_slabs_uri(slabs)?;
    let remote_cfg = RemoteStoreConfig::default();
    let (store, obj_path) =
        build_object_store(&slabs_uri, &remote_cfg).map_err(|e| e.to_string())?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;

    let (directory, zstd_dict, osv_bytes) = rt.block_on(load_osv_metadata(&store, &obj_path))?;

    let warp_cfg = if no_warp {
        WarpConfig {
            tau_req: 0.0,
            tau_byte: 1.0,
            max_coalesce_bytes: 0,
        }
    } else {
        WarpConfig::default()
    };
    let refine_stage = RefineStage::new(0, warp_cfg, zstd_dict);
    let simd = SimdEngine::new_auto();

    let mut results = Vec::new();
    for &nprobe in nprobe_list {
        for &refine_m in refine_list {
            let plan = SearchOptions {
                nprobe,
                candidates: refine_m,
                top_k,
                rerank_top_m: 0,
                refine_multiplier,
            }
            .resolve(orig_dim);

            let cache_bytes = cache_mb.saturating_mul(1024 * 1024);
            let cache = MinipageLruCache::new(cache_bytes);

            let mut latencies = Vec::with_capacity(q_count);
            let mut s1_sum = 0.0_f64;
            let mut warp_us_sum = 0.0_f64;
            let mut s23_sum = 0.0_f64;
            let mut raw_pages_sum = 0_usize;
            let mut s3_gets_sum = 0_usize;
            let mut bytes_sum = 0_u64;
            let mut recall_hits = 0_usize;
            let mut recall_denom = 0_usize;

            let wall_start = Instant::now();
            for qi in 0..q_count {
                let q = &q_flat[qi * q_dim..(qi + 1) * q_dim];
                let t0 = Instant::now();
                let t_s1 = Instant::now();
                let stage1_hits = osi
                    .search_candidates(q, plan.effective_nprobe, plan.stage1_limit, &simd)
                    .map_err(|e| e.to_string())?;
                let s1_ms = t_s1.elapsed().as_secs_f64() * 1000.0;

                let t_s23 = Instant::now();
                let (refined, stats) = rt
                    .block_on(refine_stage.refine_scored_async(
                        q,
                        &stage1_hits,
                        top_k,
                        &directory,
                        &cache,
                        &store,
                        &obj_path,
                        &remote_cfg,
                    ))
                    .map_err(|e| e.to_string())?;
                let s23_ms = t_s23.elapsed().as_secs_f64() * 1000.0;

                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
                s1_sum += s1_ms;
                warp_us_sum += stats.warp_plan_time_us as f64;
                s23_sum += s23_ms;
                raw_pages_sum += stats.raw_page_requests;
                s3_gets_sum += stats.coalesced_range_gets;
                bytes_sum += stats.bytes_fetched;

                let gt_k = top_k.min(gt_cols);
                let gt_set: HashSet<u64> = gt_flat[qi * gt_cols..qi * gt_cols + gt_k]
                    .iter()
                    .filter_map(|&v| u64::try_from(v).ok())
                    .collect();
                for hit in refined.iter().take(top_k) {
                    if gt_set.contains(&hit.doc_id) {
                        recall_hits += 1;
                    }
                }
                recall_denom += gt_k;
            }
            let total_wall_s = wall_start.elapsed().as_secs_f64();
            latencies.sort_by(f64::total_cmp);

            let mean_ms = latencies.iter().sum::<f64>() / (q_count.max(1) as f64);
            let avg_raw = (raw_pages_sum as f64) / (q_count.max(1) as f64);
            let avg_gets = (s3_gets_sum as f64) / (q_count.max(1) as f64);
            let coalesce_factor = if avg_gets > 0.0 {
                avg_raw / avg_gets
            } else {
                1.0
            };

            results.push(SearchEvalPoint {
                nprobe,
                refine_m,
                requested_nprobe: nprobe,
                effective_nprobe: plan.effective_nprobe,
                requested_candidates: refine_m,
                effective_candidates: plan.effective_candidates,
                refine_limit: plan.refine_limit,
                stage1_limit: plan.stage1_limit,
                refine_multiplier: plan.refine_multiplier,
                low_dim_oversample: plan.low_dim_oversample,
                auto_candidates: refine_m == 0,
                orig_dim,
                warp_enabled: !no_warp,
                queries_evaluated: q_count,
                recall_at_10: (recall_hits as f64) / (recall_denom.max(1) as f64),
                p50_ms: percentile(&latencies, 50),
                p95_ms: percentile(&latencies, 95),
                p99_ms: percentile(&latencies, 99),
                mean_ms,
                qps: (q_count as f64) / total_wall_s.max(1e-9),
                stage1_mean_ms: s1_sum / (q_count.max(1) as f64),
                warp_plan_mean_us: warp_us_sum / (q_count.max(1) as f64),
                stage2_3_mean_ms: s23_sum / (q_count.max(1) as f64),
                avg_raw_pages: avg_raw,
                avg_s3_gets: avg_gets,
                warp_coalesce_factor: coalesce_factor,
                avg_bytes_fetched: (bytes_sum as f64) / (q_count.max(1) as f64),
                osi_bytes,
                osv_bytes,
                vm_hwm_mb: read_vm_hwm_mb(),
                simd_tier: format!("{:?}", simd.level()),
            });
        }
    }

    if let Some(parent) = out_json.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let body = serde_json::to_string_pretty(&results).map_err(|e| e.to_string())?;
    std::fs::write(out_json, body).map_err(|e| e.to_string())?;
    Ok(())
}

fn dcg_at_k(relevances: &[f64], k: usize) -> f64 {
    relevances
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| {
            if rel > 0.0 {
                rel / ((i + 2) as f64).log2()
            } else {
                0.0
            }
        })
        .sum()
}

fn run_lutrank_eval(
    osr_path: &Path,
    queries_json: &Path,
    candidates_per_query: usize,
    top_k: usize,
    max_queries: usize,
    out_json: &Path,
) -> Result<(), String> {
    let osr_bytes = std::fs::metadata(osr_path)
        .map_err(|e| e.to_string())?
        .len();
    let osr = OsrStore::open_mmap(osr_path).map_err(|e| e.to_string())?;
    let hdr = *osr.header();
    let token_dim = usize::from(hdr.token_dim);
    let num_docs = usize::try_from(hdr.num_docs).unwrap_or(0);

    let raw_q = std::fs::read_to_string(queries_json).map_err(|e| e.to_string())?;
    let queries: Vec<BeirQuery> = serde_json::from_str(&raw_q).map_err(|e| e.to_string())?;
    let q_count = queries.len().min(max_queries);

    let simd = SimdEngine::new_auto();
    let mut lut_synth_us = Vec::with_capacity(q_count);
    struct PreparedLutQuery<'a> {
        flat_tokens: Vec<f32>,
        num_tokens: usize,
        cand_ids: Vec<u64>,
        qrels: &'a HashMap<String, i64>,
    }

    let mut prepared = Vec::with_capacity(q_count);
    for (qi, q) in queries.iter().take(q_count).enumerate() {
        let mut flat_tokens = Vec::with_capacity(q.tokens.len() * token_dim);
        for tok in &q.tokens {
            if tok.len() != token_dim {
                return Err(OspreyError::InvalidDimension { dim: tok.len() }.to_string());
            }
            flat_tokens.extend_from_slice(tok);
        }

        // Deterministic candidate set of size `candidates_per_query`:
        // includes all relevant docs for `q` plus deterministic non-relevant negatives from `.osr`.
        let mut cand_ids: Vec<u64> = Vec::with_capacity(candidates_per_query);
        let mut seen: HashSet<u64> = HashSet::with_capacity(candidates_per_query);
        for doc_str in q.qrels.keys() {
            if let Ok(did) = doc_str.parse::<u64>() {
                if seen.insert(did) {
                    cand_ids.push(did);
                }
            }
        }
        let mut cursor = ((qi * 7919) % num_docs.max(1)) as u64;
        while cand_ids.len() < candidates_per_query && cand_ids.len() < num_docs {
            if seen.insert(cursor) {
                cand_ids.push(cursor);
            }
            cursor = (cursor + 17) % (num_docs as u64);
        }
        prepared.push(PreparedLutQuery {
            flat_tokens,
            num_tokens: q.tokens.len(),
            cand_ids,
            qrels: &q.qrels,
        });
    }

    // Warmup mmap pages and CPU caches (3 queries, matching PyTorch/ONNX baselines).
    for pq in prepared.iter().take(3.min(q_count)) {
        let q_lut = QuantizedQueryLut::synthesize(
            &pq.flat_tokens,
            pq.num_tokens,
            token_dim,
            hdr.gamma,
            hdr.fwht_seed,
            &simd,
        )
        .map_err(|e| e.to_string())?;
        for &doc_id in &pq.cand_ids {
            let _ = osr
                .score_doc_id_maxsim(doc_id, &q_lut, &simd)
                .map_err(|e| e.to_string())?;
        }
    }

    let mut latencies_ms = Vec::with_capacity(q_count);
    let mut all_scored: Vec<(Vec<(u64, f32)>, &HashMap<String, i64>)> = Vec::with_capacity(q_count);

    let wall_start = Instant::now();
    for pq in &prepared {
        let t0 = Instant::now();
        let t_lut = Instant::now();
        let q_lut = QuantizedQueryLut::synthesize(
            &pq.flat_tokens,
            pq.num_tokens,
            token_dim,
            hdr.gamma,
            hdr.fwht_seed,
            &simd,
        )
        .map_err(|e| e.to_string())?;
        lut_synth_us.push(t_lut.elapsed().as_secs_f64() * 1_000_000.0);

        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(pq.cand_ids.len());
        for (idx, &doc_id) in pq.cand_ids.iter().enumerate() {
            if let Some(&next_id) = pq.cand_ids.get(idx + 1) {
                osr.prefetch_doc_id(next_id, &simd);
            }
            let sc = osr
                .score_doc_id_maxsim(doc_id, &q_lut, &simd)
                .map_err(|e| e.to_string())?
                .unwrap_or(f32::NEG_INFINITY);
            scored.push((doc_id, sc));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        all_scored.push((scored, pq.qrels));
    }
    let total_wall_s = wall_start.elapsed().as_secs_f64();

    let mut ndcg_sum = 0.0_f64;
    let mut mrr_sum = 0.0_f64;
    let mut qrel_queries = 0_usize;
    for (scored, qrels) in &all_scored {
        if !qrels.is_empty() {
            qrel_queries += 1;
            let rels: Vec<f64> = scored
                .iter()
                .take(top_k)
                .map(|(did, _)| qrels.get(&did.to_string()).copied().unwrap_or(0).max(0) as f64)
                .collect();
            let mut ideal_rels: Vec<f64> = qrels.values().map(|&v| v.max(0) as f64).collect();
            ideal_rels.sort_by(f64::total_cmp);
            ideal_rels.reverse();

            let idcg = dcg_at_k(&ideal_rels, top_k);
            if idcg > 0.0 {
                ndcg_sum += dcg_at_k(&rels, top_k) / idcg;
            }
            for (rank, &r) in rels.iter().enumerate() {
                if r > 0.0 {
                    mrr_sum += 1.0 / ((rank + 1) as f64);
                    break;
                }
            }
        }
    }
    latencies_ms.sort_by(f64::total_cmp);

    let report = LutrankEvalReport {
        osr_path: osr_path.display().to_string(),
        osr_bytes,
        simd_tier: format!("{:?}", simd.level()),
        queries_evaluated: q_count,
        candidates_per_query,
        lut_synth_mean_us: lut_synth_us.iter().sum::<f64>() / (q_count.max(1) as f64),
        p50_ms: percentile(&latencies_ms, 50),
        p95_ms: percentile(&latencies_ms, 95),
        p99_ms: percentile(&latencies_ms, 99),
        mean_ms: latencies_ms.iter().sum::<f64>() / (q_count.max(1) as f64),
        qps: (q_count as f64) / total_wall_s.max(1e-9),
        ndcg_at_10: ndcg_sum / (qrel_queries.max(1) as f64),
        mrr_at_10: mrr_sum / (qrel_queries.max(1) as f64),
        vm_hwm_mb: read_vm_hwm_mb(),
    };

    if let Some(parent) = out_json.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let body = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(out_json, body).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::SearchEval {
            osi,
            slabs,
            queries_npy,
            gt_npy,
            nprobe_list,
            refine_list,
            refine_multiplier,
            top_k,
            max_queries,
            cache_mb,
            no_warp,
            out_json,
        } => {
            let nprobes = parse_usize_list(&nprobe_list)?;
            let refines = parse_usize_list(&refine_list)?;
            run_search_eval(
                &osi,
                &slabs,
                &queries_npy,
                &gt_npy,
                &nprobes,
                &refines,
                refine_multiplier,
                top_k,
                max_queries,
                cache_mb,
                no_warp,
                &out_json,
            )
        }
        Cmd::Serve {
            osi,
            slabs,
            bind,
            cache_mb,
            no_warp,
        } => run_server(&osi, &slabs, &bind, cache_mb, no_warp),
        Cmd::LutrankEval {
            osr,
            queries_json,
            candidates_per_query,
            top_k,
            max_queries,
            out_json,
        } => run_lutrank_eval(
            &osr,
            &queries_json,
            candidates_per_query,
            top_k,
            max_queries,
            &out_json,
        ),
    }
}
