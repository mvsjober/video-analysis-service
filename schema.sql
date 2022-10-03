DROP TABLE IF EXISTS files;
CREATE TABLE files (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
  filename TEXT NOT NULL,
  content_type TEXT,
  created TEXT DEFAULT CURRENT_TIMESTAMP,
  file_id TEXT UNIQUE
);

DROP TABLE IF EXISTS analyses;
CREATE TABLE analyses (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
  task_id TEXT UNIQUE,
  file_id TEXT,
  file2_id TEXT,
  analysis_name TEXT,
  state TEXT,
  created TEXT DEFAULT CURRENT_TIMESTAMP
);
