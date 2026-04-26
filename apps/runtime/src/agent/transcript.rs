use super::Event;
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::PathBuf,
};

#[derive(Clone)]
pub struct TranscriptStore {
    root: PathBuf,
}

impl TranscriptStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn append_many(&self, run_id: &str, events: &[Event]) -> Result<(), std::io::Error> {
        fs::create_dir_all(&self.root)?;
        let path = self.path(run_id);
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        for event in events {
            let line = serde_json::to_string(event).unwrap_or_else(|_| "{}".into());
            writeln!(file, "{line}")?;
        }
        Ok(())
    }

    pub fn read(&self, run_id: &str) -> Result<Vec<Event>, std::io::Error> {
        let path = self.path(run_id);
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if let Ok(event) = serde_json::from_str(&line) {
                events.push(event);
            }
        }
        Ok(events)
    }

    fn path(&self, run_id: &str) -> PathBuf {
        let safe = run_id
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-')
            .collect::<String>();
        self.root.join(format!("{safe}.jsonl"))
    }
}
