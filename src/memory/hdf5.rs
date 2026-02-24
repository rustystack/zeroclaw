//! HDF5 memory backend — wraps EdgeHDF5 `HDF5Memory` for single-file,
//! daemon-free agent memory with hardware-adaptive vector search and BM25.

use crate::memory::traits::{Memory, MemoryCategory, MemoryEntry};
use anyhow::Context;
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use edgehdf5_memory::{
    bm25::BM25Index,
    AgentMemory, HDF5Memory,
    MemoryConfig as Hdf5Config,
    MemoryEntry as Hdf5Entry,
};

/// Default embedding dimension when none is configured.
const DEFAULT_EMBEDDING_DIM: usize = 384;

/// HDF5 agent identifier used in the .h5 file metadata.
const AGENT_ID: &str = "zeroclaw";

/// Adapter-side metadata for each entry, indexed by HDF5 numeric index.
#[derive(Clone)]
struct EntryMeta {
    key: String,
    content: String,
    category: MemoryCategory,
    session_id: Option<String>,
    timestamp: f64,
    deleted: bool,
}

/// Inner state protected by a mutex (EdgeHDF5 is sync, takes `&mut self`).
struct Inner {
    hdf5: HDF5Memory,
    bm25: BM25Index,
    /// Maps ZeroClaw string keys → HDF5 numeric indices.
    key_map: HashMap<String, usize>,
    /// Adapter-side metadata mirror, indexed by HDF5 slot index.
    entries: Vec<EntryMeta>,
    /// Chunks for BM25 indexing, parallel to entries.
    chunks: Vec<String>,
    /// Tombstone flags parallel to entries (0=active, 1=deleted).
    tombstones: Vec<u8>,
}

impl Inner {
    fn rebuild_bm25(&mut self) {
        self.bm25.rebuild(&self.chunks, &self.tombstones);
    }
}

/// HDF5-backed memory backend for ZeroClaw.
///
/// Stores all memories in a single `.h5` file with BM25 keyword search for
/// `recall()`. Thread-safe via `Arc<Mutex<Inner>>` since EdgeHDF5 requires
/// `&mut self` for writes.
pub struct Hdf5Memory {
    inner: Arc<Mutex<Inner>>,
    path: PathBuf,
}

impl Hdf5Memory {
    /// Create or open an HDF5 memory backend at `workspace_dir/brain.h5`.
    pub fn new(workspace_dir: &Path) -> anyhow::Result<Self> {
        Self::with_path(workspace_dir.join("brain.h5"))
    }

    /// Create or open an HDF5 memory backend at the given path.
    pub fn with_path(path: PathBuf) -> anyhow::Result<Self> {
        let hdf5 = if path.exists() && std::fs::metadata(&path).map_or(false, |m| m.len() > 0) {
            HDF5Memory::open(&path).with_context(|| {
                format!("failed to open HDF5 memory at {}", path.display())
            })?
        } else {
            let config = Hdf5Config::new(path.clone(), AGENT_ID, DEFAULT_EMBEDDING_DIM);
            HDF5Memory::create(config).with_context(|| {
                format!("failed to create HDF5 memory at {}", path.display())
            })?
        };

        // Hydrate in-memory state from existing HDF5 data.
        let chunks = hdf5.cache.chunks.clone();
        let tombstones = hdf5.cache.tombstones.clone();
        let bm25 = BM25Index::build(&chunks, &tombstones);

        let mut entries: Vec<EntryMeta> = Vec::with_capacity(chunks.len());
        let mut key_map: HashMap<String, usize> = HashMap::with_capacity(chunks.len());

        for i in 0..chunks.len() {
            let key = hdf5.cache.source_channels.get(i).cloned().unwrap_or_default();
            let tag = hdf5.cache.tags.get(i).cloned().unwrap_or_default();
            let ts = hdf5.cache.timestamps.get(i).copied().unwrap_or(0.0);
            let sid = hdf5.cache.session_ids.get(i).cloned().unwrap_or_default();
            let deleted = tombstones.get(i).copied().unwrap_or(0) != 0;

            if !deleted && !key.is_empty() {
                key_map.insert(key.clone(), i);
            }

            entries.push(EntryMeta {
                key,
                content: chunks.get(i).cloned().unwrap_or_default(),
                category: tag_to_category(&tag),
                timestamp: ts,
                session_id: if sid.is_empty() { None } else { Some(sid) },
                deleted,
            });
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                hdf5,
                bm25,
                key_map,
                entries,
                chunks,
                tombstones,
            })),
            path,
        })
    }
}

/// Convert a `MemoryCategory` to a tag string for HDF5 storage.
fn category_to_tag(cat: &MemoryCategory) -> String {
    format!("cat:{cat}")
}

/// Parse a tag string back into a `MemoryCategory`.
fn tag_to_category(tag: &str) -> MemoryCategory {
    let raw = tag.strip_prefix("cat:").unwrap_or(tag);
    match raw {
        "core" => MemoryCategory::Core,
        "daily" => MemoryCategory::Daily,
        "conversation" => MemoryCategory::Conversation,
        other => MemoryCategory::Custom(other.to_string()),
    }
}

/// Build a `MemoryEntry` from adapter-side metadata.
fn meta_to_entry(idx: usize, meta: &EntryMeta) -> MemoryEntry {
    MemoryEntry {
        id: idx.to_string(),
        key: meta.key.clone(),
        content: meta.content.clone(),
        category: meta.category.clone(),
        timestamp: format_timestamp(meta.timestamp),
        session_id: meta.session_id.clone(),
        score: None,
    }
}

/// Format a UNIX timestamp (f64) as an ISO 8601 string.
fn format_timestamp(ts: f64) -> String {
    use chrono::{TimeZone, Utc};
    #[allow(clippy::cast_possible_truncation)]
    let secs = ts as i64;
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let nanos = ((ts - secs as f64) * 1_000_000_000.0) as u32;
    Utc.timestamp_opt(secs, nanos)
        .single()
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_else(|| ts.to_string())
}

/// Get the current UNIX timestamp as f64.
fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[async_trait]
impl Memory for Hdf5Memory {
    fn name(&self) -> &str {
        "hdf5"
    }

    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let inner = Arc::clone(&self.inner);
        let key = key.to_string();
        let content = content.to_string();
        let session = session_id.map(str::to_string);
        let tag = category_to_tag(&category);
        let ts = now_timestamp();

        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock();

            // If key already exists, delete the old entry first.
            if let Some(&old_idx) = guard.key_map.get(&key) {
                let _ = guard.hdf5.delete(old_idx);
                if old_idx < guard.tombstones.len() {
                    guard.tombstones[old_idx] = 1;
                    guard.entries[old_idx].deleted = true;
                }
            }

            let hdf5_entry = Hdf5Entry {
                chunk: content.clone(),
                embedding: Vec::new(),
                source_channel: key.clone(),
                timestamp: ts,
                session_id: session.clone().unwrap_or_default(),
                tags: tag,
            };

            let idx = guard
                .hdf5
                .save(hdf5_entry)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            // Grow adapter-side vectors to match.
            while guard.entries.len() <= idx {
                guard.entries.push(EntryMeta {
                    key: String::new(),
                    content: String::new(),
                    category: MemoryCategory::Core,
                    session_id: None,
                    timestamp: 0.0,
                    deleted: true,
                });
                guard.chunks.push(String::new());
                guard.tombstones.push(1);
            }

            guard.entries[idx] = EntryMeta {
                key: key.clone(),
                content: content.clone(),
                category: category.clone(),
                session_id: session,
                timestamp: ts,
                deleted: false,
            };
            guard.chunks[idx] = content;
            guard.tombstones[idx] = 0;
            guard.key_map.insert(key, idx);
            guard.rebuild_bm25();
            Ok(())
        })
        .await
        .context("spawn_blocking join failed")?
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        let inner = Arc::clone(&self.inner);
        let query = query.to_string();
        let session = session_id.map(str::to_string);

        tokio::task::spawn_blocking(move || {
            let guard = inner.lock();
            // Over-fetch to account for session filtering.
            let results = guard.bm25.search(&query, limit * 2);
            let mut entries = Vec::with_capacity(limit);

            for (idx, score) in results {
                if entries.len() >= limit {
                    break;
                }
                if idx >= guard.entries.len() || guard.entries[idx].deleted {
                    continue;
                }
                let meta = &guard.entries[idx];
                // Session filter.
                if let Some(ref sid) = session {
                    if let Some(ref entry_sid) = meta.session_id {
                        if entry_sid != sid {
                            continue;
                        }
                    }
                }
                let mut entry = meta_to_entry(idx, meta);
                entry.score = Some(f64::from(score));
                entries.push(entry);
            }
            Ok(entries)
        })
        .await
        .context("spawn_blocking join failed")?
    }

    async fn get(&self, key: &str) -> anyhow::Result<Option<MemoryEntry>> {
        let inner = Arc::clone(&self.inner);
        let key = key.to_string();

        tokio::task::spawn_blocking(move || {
            let guard = inner.lock();
            let Some(&idx) = guard.key_map.get(&key) else {
                return Ok(None);
            };
            if idx >= guard.entries.len() || guard.entries[idx].deleted {
                return Ok(None);
            }
            Ok(Some(meta_to_entry(idx, &guard.entries[idx])))
        })
        .await
        .context("spawn_blocking join failed")?
    }

    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        session_id: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        let inner = Arc::clone(&self.inner);
        let cat = category.cloned();
        let session = session_id.map(str::to_string);

        tokio::task::spawn_blocking(move || {
            let guard = inner.lock();
            let mut entries = Vec::new();

            for (i, meta) in guard.entries.iter().enumerate() {
                if meta.deleted {
                    continue;
                }
                if let Some(ref c) = cat {
                    if meta.category != *c {
                        continue;
                    }
                }
                if let Some(ref sid) = session {
                    if let Some(ref entry_sid) = meta.session_id {
                        if entry_sid != sid {
                            continue;
                        }
                    }
                }
                entries.push(meta_to_entry(i, meta));
            }
            Ok(entries)
        })
        .await
        .context("spawn_blocking join failed")?
    }

    async fn forget(&self, key: &str) -> anyhow::Result<bool> {
        let inner = Arc::clone(&self.inner);
        let key = key.to_string();

        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock();
            let Some(&idx) = guard.key_map.get(&key) else {
                return Ok(false);
            };
            guard
                .hdf5
                .delete(idx)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            if idx < guard.tombstones.len() {
                guard.tombstones[idx] = 1;
                guard.entries[idx].deleted = true;
            }
            guard.key_map.remove(&key);
            guard.rebuild_bm25();
            Ok(true)
        })
        .await
        .context("spawn_blocking join failed")?
    }

    async fn count(&self) -> anyhow::Result<usize> {
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || {
            let guard = inner.lock();
            Ok(guard.hdf5.count_active())
        })
        .await
        .context("spawn_blocking join failed")?
    }

    async fn health_check(&self) -> bool {
        self.path.exists() && self.path.is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_backend(tmp: &TempDir) -> Hdf5Memory {
        Hdf5Memory::new(tmp.path()).expect("failed to create HDF5 backend")
    }

    #[tokio::test]
    async fn store_and_get_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);

        mem.store("lang", "Rust", MemoryCategory::Core, None)
            .await
            .unwrap();

        let entry = mem.get("lang").await.unwrap().expect("entry not found");
        assert_eq!(entry.key, "lang");
        assert_eq!(entry.content, "Rust");
        assert_eq!(entry.category, MemoryCategory::Core);
    }

    #[tokio::test]
    async fn store_overwrites_existing_key() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);

        mem.store("lang", "Python", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("lang", "Rust", MemoryCategory::Core, None)
            .await
            .unwrap();

        let entry = mem.get("lang").await.unwrap().unwrap();
        assert_eq!(entry.content, "Rust");
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn recall_finds_matching_entries() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);

        mem.store(
            "fact1",
            "Rust is a systems programming language",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        mem.store(
            "fact2",
            "Python is a scripting language",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();

        let results = mem.recall("Rust programming", 10, None).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].key, "fact1");
        assert!(results[0].score.unwrap() > 0.0);
    }

    #[tokio::test]
    async fn recall_with_session_filter() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);

        mem.store("a", "hello world", MemoryCategory::Conversation, Some("s1"))
            .await
            .unwrap();
        mem.store(
            "b",
            "hello universe",
            MemoryCategory::Conversation,
            Some("s2"),
        )
        .await
        .unwrap();

        let results = mem.recall("hello", 10, Some("s1")).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "a");
    }

    #[tokio::test]
    async fn forget_removes_entry() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);

        mem.store("key1", "value1", MemoryCategory::Core, None)
            .await
            .unwrap();
        assert!(mem.forget("key1").await.unwrap());
        assert!(mem.get("key1").await.unwrap().is_none());
        assert_eq!(mem.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn forget_returns_false_for_missing_key() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);
        assert!(!mem.forget("nonexistent").await.unwrap());
    }

    #[tokio::test]
    async fn list_filters_by_category() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);

        mem.store("a", "core fact", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "daily log", MemoryCategory::Daily, None)
            .await
            .unwrap();

        let core = mem.list(Some(&MemoryCategory::Core), None).await.unwrap();
        assert_eq!(core.len(), 1);
        assert_eq!(core[0].key, "a");

        let all = mem.list(None, None).await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn count_tracks_active_entries() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);

        assert_eq!(mem.count().await.unwrap(), 0);
        mem.store("a", "val", MemoryCategory::Core, None)
            .await
            .unwrap();
        assert_eq!(mem.count().await.unwrap(), 1);
        mem.store("b", "val", MemoryCategory::Core, None)
            .await
            .unwrap();
        assert_eq!(mem.count().await.unwrap(), 2);
        mem.forget("a").await.unwrap();
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn health_check_passes_for_existing_file() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);
        assert!(mem.health_check().await);
    }

    #[tokio::test]
    async fn name_returns_hdf5() {
        let tmp = TempDir::new().unwrap();
        let mem = make_backend(&tmp);
        assert_eq!(mem.name(), "hdf5");
    }
}
