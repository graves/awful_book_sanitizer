//! # awful_book_sanitizer
//!
//! A command-line tool for cleaning up OCR’d book excerpts from `.txt` files.
//!
//! Many scanned books contain corrupted characters, misspelled words, and poor grammar.
//! This tool reads plain text files, splits them into 500-token chunks, and asks a
//! Large Language Model (LLM) (via an OpenAI-compatible endpoint) to **sanitize** them.
//!
//! The sanitized chunks are appended into YAML files named after the corresponding
//! input `.txt` file. Each run produces YAML like:
//!
//! ```yaml
//! chunks:
//!   - |-
//!     Cleaned text line 1
//!     Cleaned text line 2
//! ```
//!
//! ## Multi-endpoint concurrency
//!
//! You can specify multiple configuration files (`--config` flags), each of which
//! points to a separate LLM backend (e.g., a local instance, a cloud endpoint).
//! The tool spawns **one worker thread per configuration file**, allowing multiple
//! sanitizers to run concurrently across different endpoints.
//!
//! ## Example
//! ```bash
//! awful_book_sanitizer \
//!   --input /path/to/input \
//!   --output /path/to/output \
//!   --config llama.yaml colab.yaml
//! ```
//!
//! This will:
//! - Load text files from `/path/to/input`.
//! - Spawn two threads, one using `llama.yaml`, the other `colab.yaml`.
//! - Write output YAMLs under `/path/to/output`.

use std::path::PathBuf;
use std::{fs, time::Duration};

use awful_aj::{
    api::ask,
    config::AwfulJadeConfig,
    template::{self, ChatTemplate},
};
use clap::{Parser, command};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::Write as IoWrite;
use text_splitter::{ChunkConfig, TextSplitter};
use tiktoken_rs::cl100k_base;
use tokio::task::spawn_blocking;
use tokio::time::sleep;

/// Command-line arguments for `awful_book_sanitizer`.
#[derive(Parser, Debug)]
#[command(name = "awful_book_sanitizer")]
#[command(about = "Clean up excerpts from books formatted as txt", long_about = None)]
struct Args {
    /// Path to directory of `.txt` files to sanitize.
    #[arg(short, long = "input")]
    input_dir: PathBuf,

    /// Path to directory where `.yaml` files will be written.
    #[arg(short, long = "output")]
    output_dir: PathBuf,

    /// One or more configuration files specifying API endpoints.
    ///
    /// Each file is parsed into an [`AwfulJadeConfig`] and run in its own worker.
    #[arg(long = "config", num_args = 1..)]
    config: Vec<PathBuf>,
}

/// Data structure for sanitized book excerpts, returned by the model.
///
/// Each LLM response is expected to be valid JSON with this shape.
#[derive(Debug, Deserialize, Serialize)]
pub struct BookChunk {
    /// The sanitized text excerpt (cleaned up by the model).
    pub sanitizedBookExcerpt: String,
}

/// Entry point: parses arguments, spawns worker tasks, and drives sanitization.
///
/// For each `--config` file, a separate blocking worker thread is spawned, running
/// [`process_files`]. All workers run in parallel.
///
/// Returns `Ok(())` on success; prints errors to stderr otherwise.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Parse command-line arguments
    let args = Args::parse();

    // Clone paths to avoid moving them during spawning
    let input_dir_path = args.input_dir.clone();
    let output_dir_path = args.output_dir.to_str().unwrap().to_string();

    // Spawn tasks for each configuration file
    let mut handles = Vec::new();

    for config_path in &args.config {
        // Load configuration from file
        let config = awful_aj::config::load_config(config_path.to_str().unwrap())
            .map_err(|e| format!("Config load error: {e}"))?;

        // Clone paths for safe use in spawned tasks
        let input_clone = input_dir_path.clone();
        let output_clone = output_dir_path.clone();

        // Spawn a blocking task to process files
        handles.push(spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                process_files(&input_clone, &output_clone, config)
                    .await
                    .map_err(|e| -> Box<dyn Error + Send + Sync> { e.into() })
            })
        }));
    }

    // Wait for all tasks to complete
    for handle in handles {
        if let Err(e) = handle.await? {
            eprintln!("Error in task: {}", e);
        }
    }

    Ok(())
}

/// Process `.txt` files under the given directory and sanitize their contents.
///
/// - Splits each file into ~500-token chunks.
/// - Submits each chunk to the model using [`fetch_with_backoff`].
/// - Appends sanitized chunks to a YAML file named after the input file.
///
/// # Parameters
/// - `input_dir`: Path to directory containing `.txt` files.
/// - `output_dir_path`: Path where YAML files are written.
/// - `config`: Configuration for model endpoint.
///
/// # Errors
/// Returns `Err(String)` on filesystem, config, or API errors. Errors for
/// individual files/chunks are logged and do not abort other files.
///
/// # Example
/// ```no_run
/// # async fn demo(cfg: awful_aj::config::AwfulJadeConfig) {
/// let res = process_files(&"/tmp/books".into(), "/tmp/out", cfg).await;
/// if let Err(err) = res {
///     eprintln!("Sanitization failed: {err}");
/// }
/// # }
/// ```
async fn process_files(
    input_dir: &PathBuf,
    output_dir_path: &str,
    config: AwfulJadeConfig,
) -> Result<(), String> {
    // Initialize tokenizer for tokenization
    let tokenizer = cl100k_base().map_err(|e| e.to_string())?;
    let max_tokens = 500;

    // Configure text splitter to chunk content
    let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));

    // Load template for sanitization
    let template = template::load_template("book_txt_sanitizer")
        .await
        .map_err(|e| format!("Template load error: {e}"))?;

    // Process each file in the input directory
    for entry in fs::read_dir(input_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = &entry.path();

        // Check if the file is a `.txt` text file
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let filename = path.file_name().unwrap().to_string_lossy();
            let mut yaml_path = format!("{}/{}.yaml", output_dir_path, filename);

            // Open YAML file for writing
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&yaml_path)
                .map_err(|e| e.to_string())?;

            // Write YAML header
            writeln!(file, "chunks:").map_err(|e| e.to_string())?;

            // Read and process the text content
            let contents = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let chunks: Vec<_> = splitter.chunks(&contents).collect();

            // Process each chunk
            for chunk in chunks {
                let book_chunk = fetch_with_backoff(&config, &chunk, &template)
                    .await
                    .map_err(|e| e.to_string())?;

                if let Some(sanitized_text) = book_chunk {
                    // Write sanitized content to YAML
                    write_row_to_file(sanitized_text, &mut yaml_path).map_err(|e| e.to_string())?;
                }
            }
        }
    }

    Ok(())
}

/// Append a sanitized text chunk to an output YAML file.
///
/// Each chunk is written as:
/// ```yaml
///   - |-
///     line 1
///     line 2
/// ```
///
/// # Parameters
/// - `chunk`: Sanitized text to append.
/// - `yaml_path`: Path to YAML file (modified by reference).
///
/// # Errors
/// Returns any I/O or formatting errors encountered.
///
/// # Example
/// ```no_run
/// let mut yaml_path = "/tmp/out/book.yaml".to_string();
/// write_row_to_file("Cleaned text".into(), &mut yaml_path).unwrap();
/// ```
pub fn write_row_to_file(
    chunk: String,
    yaml_path: &mut String,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&yaml_path)?; // Open YAML file for appending

    // Write YAML line with content
    writeln!(file, "\t- |-").map_err(|e| e.to_string())?;
    for line in chunk.lines() {
        writeln!(file, "\t\t{}", line).map_err(|e| e.to_string())?;
    }

    Ok(())
}

// The maximum number of times to retry a request to the LLM service.
const MAX_RETRIES: u32 = 5;
// The initial delay between retries that grows exponentially.
const BASE_DELAY_MS: u64 = 500;

/// Call the model to sanitize a text chunk with exponential backoff.
///
/// Retries failed API requests up to [`MAX_RETRIES`] times, waiting
/// `BASE_DELAY_MS * 2^attempt` ms between tries.
///
/// # Parameters
/// - `config`: Model configuration (endpoint, key, etc.).
/// - `chunk`: Text chunk to sanitize.
/// - `template`: Sanitization prompt template.
///
/// # Returns
/// - `Ok(Some(cleaned_text))` if successful.
/// - `Ok(None)` if the API responded with an empty result (`"{}"`).
/// - `Err` if all retries failed or response parse failed.
///
/// # Example
/// ```no_run
/// # async fn demo(cfg: awful_aj::config::AwfulJadeConfig, t: awful_aj::template::ChatTemplate) {
/// if let Ok(Some(text)) = fetch_with_backoff(&cfg, "raw text", &t).await {
///     println!("Sanitized: {text}");
/// }
/// # }
/// ```
async fn fetch_with_backoff(
    config: &AwfulJadeConfig,
    chunk: &str,
    template: &ChatTemplate,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    for attempt in 0..=MAX_RETRIES {
        let res = ask(config, chunk.to_string(), template, None, None).await;

        match res {
            Ok(res) => {
                if res != "{}" {
                    let book_chunk: BookChunk = serde_json::from_str(&res)?;
                    return Ok(Some(book_chunk.sanitizedBookExcerpt));
                } else {
                    return Ok(None);
                }
            }
            Err(err) => {
                eprintln!("Request failed: {}", err);
            }
        }

        if attempt < MAX_RETRIES {
            let backoff = BASE_DELAY_MS * (2u64.pow(attempt));
            eprintln!("Retrying in {}ms...", backoff);
            sleep(Duration::from_millis(backoff)).await;
        }
    }

    Err("All retries failed".into())
}
