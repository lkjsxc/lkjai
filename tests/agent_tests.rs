use lkjai::agent::{event, TranscriptStore};

#[test]
fn transcript_round_trips_events() {
    let root = std::env::temp_dir().join(format!("lkjai-test-{}", uuid::Uuid::new_v4()));
    let store = TranscriptStore::new(root.clone());
    let events = vec![event("user", "hello".into(), None)];
    store.append_many("run-1", &events).unwrap();
    let loaded = store.read("run-1").unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].kind, "user");
    assert_eq!(loaded[0].content, "hello");
    std::fs::remove_dir_all(root).unwrap();
}
